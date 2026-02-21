import os
import uuid
import torch
import boto3
import tempfile
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header, Depends, status
from pydantic import BaseModel
from fastembed import ImageEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import mysql.connector
import hashlib

load_dotenv()

# --- KONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PREFIX = os.getenv("R2_PREFIX")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
INDEX_API_KEY = os.getenv("INDEX_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")



# Router definieren
# Router definieren
router = APIRouter()

async def verify_index_key(x_api_key: str = Header(..., description="API Key für Indexierungs-Zugriff")):
    if not INDEX_API_KEY:
        # Falls kein Key konfiguriert ist, loggen wir das oder failen. 
        # Sicherheitshalber failen wir, wenn der User Schutz will aber keinen Key gesetzt hat.
        print("WARNUNG: INDEX_API_KEY nicht gesetzt!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server Konfigurationsfehler: INDEX_API_KEY fehlt"
        )
        
    if x_api_key != INDEX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# Globale Variablen für Modul-Level Caching (Optional, wird hier lazy geladen)
_model_img = None
_s3_client = None
_qdrant_client = None

def get_model():
    global _model_img
    if _model_img is None:
        print(f"Lade CLIP IMAGE Model (fastembed)...")
        _model_img = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    return _model_img

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
            print("WARNUNG: R2 Credentials fehlen in der .env Datei!")
            return None
        _s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'
        )
    return _s3_client

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if not _qdrant_client.collection_exists(COLLECTION_NAME):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        # Payload Index sicherstellen
        try:
            _qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="filename",
                field_schema="keyword",
            )
        except:
            pass # Index existiert wohl schon
    return _qdrant_client

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def fetch_all_metadata():
    """
    Fetches all NID and Delta values from MySQL and returns a dict:
    { "filename": {"nid": 123, "delta": 0}, ... }
    """
    print("Fetching metadata from MySQL...")
    conn = get_db_connection()
    if not conn:
        return {}
    
    metadata_map = {}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT filename, nid, delta FROM totenbilder_bilder")
        results = cursor.fetchall()
        for row in results:
            metadata_map[row['filename']] = {"nid": row['nid'], "delta": row['delta']}
        print(f"Loaded metadata for {len(metadata_map)} images.")
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            
    return metadata_map

def process_indexing(force_reindex: bool = False, recreate_collection: bool = False):
    """
    Hintergrund-Funktion, die den R2 Bucket durchläuft und Bilder indexiert.
    """
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    model = get_model()

    if not s3 or not qdrant or not model:
        print("Fehler: Clients konnten nicht initialisiert werden.")
        return

    # 0. Metadata laden (für alle Bilder)
    metadata_map = fetch_all_metadata()

    # 1. Collection neu erstellen?
    if recreate_collection:
        print(f"!!! ACHTUNG: Lösche und erstelle Collection '{COLLECTION_NAME}' neu...")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        # Payload Index sofort wieder anlegen
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="filename",
            field_schema="keyword",
        )
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="nid", 
            field_schema="integer"
        )
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="delta", 
            field_schema="integer"
        )
        force_reindex = True # Logischerweise müssen wir dann alles neu machen

    print(f"--- Starte Indexierung. Bucket: {R2_BUCKET_NAME} (Ordner: {R2_PREFIX}) ---")
    points_buffer = []
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=R2_PREFIX)
    
    count_processed = 0
    count_skipped = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Ordner selbst überspringen
            if key == R2_PREFIX: 
                continue
            if not key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            try:
                # Prüfung: Ist das Bild schon indexiert?
                if not force_reindex:
                    res = qdrant.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(
                            must=[FieldCondition(key="filename", match=MatchValue(value=key))]
                        ),
                        limit=1,
                        with_payload=False,
                        with_vectors=False
                    )
                    if len(res[0]) > 0:
                        count_skipped += 1
                        continue

                print(f"Verarbeite: {key}...")
                
                # Bild aus R2 laden
                file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
                file_content = file_obj['Body'].read()
                
                # FastEmbed benötigt Datei-Pfad (oder Liste davon)
                file_ext = os.path.splitext(key)[1]
                vector = []
                
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
                try:
                    # Embedden
                    vector = list(model.embed([tmp_path]))[0].tolist()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                # Deterministic UUID generation
                # Using UUID5 with DNS namespace + unique key (filename) ensures 
                # same filename always gets same UUID.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

                # Payload zusammenbauen
                payload = {"filename": key}
                
                # Metadata aus MySQL mergen
                # Wir entfernen den Prefix "totenbilder/" aus dem Key für den DB-Lookup falls nötig?
                # In update_payload.py war full_key = prefix + filename. 
                # In DB steht nur "filename" (z.B. "123.jpg") oder mit pfad?
                # Check update_payload.py Zeile 93: "WHERE filename = %s".
                # Wenn in DB nur "123.jpg" steht, aber key "totenbilder/123.jpg" ist...
                # Wir sollten beides probieren oder annehmen es passt.
                # In update_payload.py Zeile 53 wird full_key gebaut. Also DB hat wohl nur den Dateinamen ohne Prefix?
                # Nein, warte: process_single(filename) nimmt filename als Argument.
                # process_all holt filename aus DB.
                # update_qdrant_point baut full_key = R2_PREFIX + filename.
                # Das impliziert: DB hat "123.jpg", Qdrant/R2 hat "totenbilder/123.jpg".
                # Also müssen wir den Prefix abschneiden für den Lookup.
                
                db_filename = key.replace(R2_PREFIX, "")
                # Fallback: Falls R2_PREFIX nicht leer ist und key damit anfängt
                
                if db_filename in metadata_map:
                    payload["nid"] = metadata_map[db_filename]["nid"]
                    payload["delta"] = metadata_map[db_filename]["delta"]
                
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points_buffer.append(point)
                count_processed += 1
                
                # Batch-Upload
                if len(points_buffer) >= 50:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
                    points_buffer = []
                    print(f"Fortschritt: {count_processed} Bilder neu verarbeitet...")
                    
            except Exception as e:
                print(f"Fehler bei {key}: {e}")

    if points_buffer:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
    
    print(f"--- Indexierung Fertig! Neu: {count_processed}, Übersprungen: {count_skipped} ---")

class SingleIndexRequest(BaseModel):
    filename: str

@router.post("/index-one", dependencies=[Depends(verify_index_key)])
async def index_single_image(request: SingleIndexRequest):
    """
    Indiziert ein einzelnes Bild aus dem S3 Bucket.
    """
    key = request.filename
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    model = get_model()

    if not s3 or not qdrant or not model:
        raise HTTPException(status_code=500, detail="Interne Services nicht verfügbar.")

    try:
        print(f"Indexiere einzelnes Bild: {key}...")
        
        # Bild aus R2 laden
        # Bild aus R2 laden
        try:
            file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            file_content = file_obj['Body'].read()
            
            file_ext = os.path.splitext(key)[1]
            vector = []
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            try:
                # Embedden
                vector = list(model.embed([tmp_path]))[0].tolist()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            print(f"Fehler beim Laden/Embedden von {key}: {e}")
            raise HTTPException(status_code=404, detail=f"Bild '{key}' konnte nicht verarbeitet werden: {e}")
        
        # Deterministic UUID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

        payload = {"filename": key}
        
        # Metadata fetch (Quick & Dirty: Einfach DB fragen für dieses eine File)
        # Oder wir nutzen fetch_all_metadata() nicht, sondern eine kleine Helper funktion?
        # Egal, wir machen es inline oder nutzen eine neue Helper funktion?
        # Da wir schon get_db_connection haben:
        
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                # DB filename check (remove prefix AND keep prefix to match both legacy and new rows)
                db_filename_stripped = key.replace(R2_PREFIX, "")
                cursor.execute("SELECT nid, delta FROM totenbilder_bilder WHERE filename = %s OR filename = %s", (db_filename_stripped, key))
                row = cursor.fetchone()
                if row:
                    payload["nid"] = row['nid']
                    payload["delta"] = row['delta']
                else:
                    print(f"WARNUNG: Keine Metadaten für {key} (stripped: {db_filename_stripped}) in totenbilder_bilder gefunden!")
                    
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error fetching metadata for single image: {e}")

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        # Upsert (direkt, ohne Buffer)
        qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        
        return {"message": f"Bild '{key}' erfolgreich indexiert.", "filename": key}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Fehler bei {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class IndexRequest(BaseModel):
    force_reindex: bool = False
    recreate_collection: bool = False

@router.post("/index", dependencies=[Depends(verify_index_key)])
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Startet die Indexierung der Bilder vom R2 Bucket in Qdrant im Hintergrund.
    Optional: recreate_collection=True löscht alles vorher!
    """
    background_tasks.add_task(process_indexing, request.force_reindex, request.recreate_collection)
    return {"message": "Indexierung wurde im Hintergrund gestartet.", "bucket": R2_BUCKET_NAME, "recreate": request.recreate_collection}
