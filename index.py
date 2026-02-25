import os
import uuid
import torch
import boto3
import tempfile
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header, Depends, status
from pydantic import BaseModel
from fastembed import ImageEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PointVectors
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
_model_txt = None
_s3_client = None
_qdrant_client = None
def get_model():
    global _model_img
    if _model_img is None:
        print(f"Lade CLIP IMAGE Model (fastembed)...")
        _model_img = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    return _model_img

def get_text_model():
    global _model_txt
    if _model_txt is None:
        print(f"Lade MiniLM TEXT Model (fastembed)...")
        _model_txt = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _model_txt

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
    Fetches all NID, Delta and Fulltext values from MySQL and returns a dict:
    { "filename": {"nid": 123, "delta": 0, "fulltext": "..." }, ... }
    """
    print("Fetching metadata from MySQL...")
    conn = get_db_connection()
    if not conn:
        return {}
    
    metadata_map = {}
    try:
        cursor = conn.cursor(dictionary=True)
        # JOIN table "totenbilder" to get Fulltext
        cursor.execute("SELECT b.filename, b.nid, b.delta, t.Fulltext FROM totenbilder_bilder b LEFT JOIN totenbilder t ON b.nid = t.nid")
        results = cursor.fetchall()
        for row in results:
            metadata_map[row['filename']] = {
                "nid": row['nid'], 
                "delta": row['delta'],
                "fulltext": row['Fulltext']
            }
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
    model_img = get_model()
    model_txt = get_text_model()

    if not s3 or not qdrant or not model_img or not model_txt:
        print("Fehler: Clients konnten nicht initialisiert werden.")
        return

    # 0. Metadata laden (für alle Bilder)
    metadata_map = fetch_all_metadata()

    # 1. Collection neu erstellen?
    if recreate_collection:
        print(f"!!! ACHTUNG: Lösche und erstelle Collection '{COLLECTION_NAME}' neu...")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "": VectorParams(size=512, distance=Distance.COSINE),
                "text": VectorParams(size=384, distance=Distance.COSINE)
            },
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
                    # Embedden Image
                    img_vector = list(model_img.embed([tmp_path]))[0].tolist()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                # Deterministic UUID generation
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

                # Payload zusammenbauen
                payload = {"filename": key}
                db_filename = key.replace(R2_PREFIX, "")
                
                # Check for Text Embedding based on delta
                text_vector = None
                if db_filename in metadata_map:
                    payload["nid"] = metadata_map[db_filename]["nid"]
                    payload["delta"] = metadata_map[db_filename]["delta"]
                    
                    if payload["delta"] == 0 and metadata_map[db_filename]["fulltext"]:
                        try:
                            text_vector = list(model_txt.embed([metadata_map[db_filename]["fulltext"]]))[0].tolist()
                        except Exception as text_e:
                            print(f"WARNUNG: Text Embedding fehlgeschlagen für {db_filename}: {text_e}")
                
                # Vector dict with named vectors
                vector_payload = img_vector # fallback
                if text_vector:
                    vector_payload = {
                        "": img_vector,
                        "text": text_vector
                    }
                
                point = PointStruct(
                    id=point_id,
                    vector=vector_payload,
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
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    model_img = get_model()
    model_txt = get_text_model()

    if not s3 or not qdrant or not model_img or not model_txt:
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
                # Embedden Image
                img_vector = list(model_img.embed([tmp_path]))[0].tolist()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            print(f"Fehler beim Laden/Embedden von {key}: {e}")
            raise HTTPException(status_code=404, detail=f"Bild '{key}' konnte nicht verarbeitet werden: {e}")
        
        # Deterministic UUID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

        payload = {"filename": key}
        
        # Metadata fetch
        text_vector = None
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                db_filename_stripped = key.replace(R2_PREFIX, "")
                cursor.execute(
                    "SELECT b.nid, b.delta, t.Fulltext FROM totenbilder_bilder b LEFT JOIN totenbilder t ON b.nid=t.nid WHERE b.filename = %s OR b.filename = %s", 
                    (db_filename_stripped, key)
                )
                row = cursor.fetchone()
                if row:
                    payload["nid"] = row['nid']
                    payload["delta"] = row['delta']
                    
                    if payload["delta"] == 0 and row['Fulltext']:
                        try:
                            text_vector = list(model_txt.embed([row['Fulltext']]))[0].tolist()
                        except Exception as text_e:
                            print(f"WARNUNG: Text Embedding fehlgeschlagen für {key}: {text_e}")
                else:
                    print(f"WARNUNG: Keine Metadaten für {key} gefunden!")
                    
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error fetching metadata for single image: {e}")

        # Vector dict with named vectors
        vector_payload = img_vector # fallback
        if text_vector:
            vector_payload = {
                "": img_vector,
                "text": text_vector
            }

        point = PointStruct(
            id=point_id,
            vector=vector_payload,
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

class UpdateTextRequest(BaseModel):
    filename: str

@router.post("/update-text-one", dependencies=[Depends(verify_index_key)])
async def update_single_text(request: UpdateTextRequest):
    """
    Indiziert den Textvektor eines Bildes (nur bei delta=0). Der Hauptvektor bleibt unberührt.
    """
    key = request.filename
    qdrant = get_qdrant_client()
    model_txt = get_text_model()

    if not qdrant or not model_txt:
        raise HTTPException(status_code=500, detail="Interne Services nicht verfügbar.")

    try:
        print(f"Update Textvektor für: {key}...")
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
        
        # Metadata fetch
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="DB Error")
            
        cursor = conn.cursor(dictionary=True)
        db_filename_stripped = key.replace(R2_PREFIX, "")
        cursor.execute(
            "SELECT b.delta, t.Fulltext FROM totenbilder_bilder b LEFT JOIN totenbilder t ON b.nid=t.nid WHERE b.filename = %s OR b.filename = %s", 
            (db_filename_stripped, key)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Eintrag in DB nicht gefunden.")
            
        if row['delta'] != 0:
            return {"message": f"Übersprungen: {key} ist eine Rückseite (delta>0).", "filename": key}
            
        fulltext = row['Fulltext']
        if not fulltext:
            return {"message": f"Übersprungen: {key} hat keinen Fulltext.", "filename": key}

        # Embedden
        text_vector = list(model_txt.embed([fulltext]))[0].tolist()
        
        # update_vectors
        qdrant.update_vectors(
            collection_name=COLLECTION_NAME,
            points=[
                PointVectors(
                    id=point_id,
                    vector={
                        "text": text_vector
                    }
                )
            ]
        )
        
        return {"message": f"Textvektor für '{key}' erfolgreich aktualisiert.", "filename": key}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Fehler bei {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_update_all_text():
    qdrant = get_qdrant_client()
    model_txt = get_text_model()
    
    if not qdrant or not model_txt:
        print("Fehler: Clients nicht verfügbar")
        return
        
    print("Starte Update aller Textvektoren...")
    
    metadata_map = fetch_all_metadata()
    count_success = 0
    count_skipped = 0
    count_error = 0
    
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["filename", "delta"],
            with_vectors=False,
            limit=500,
            offset=offset
        )
        
        for point in points:
            filename = point.payload.get("filename")
            delta = point.payload.get("delta", 0)
            
            if not filename or delta != 0:
                count_skipped += 1
                continue
                
            db_filename = filename.replace(R2_PREFIX, "")
            
            if db_filename in metadata_map:
                fulltext = metadata_map[db_filename]["fulltext"]
                if fulltext:
                    try:
                        text_vector = list(model_txt.embed([fulltext]))[0].tolist()
                        qdrant.update_vectors(
                            collection_name=COLLECTION_NAME,
                            points=[
                                PointVectors(
                                    id=point.id,
                                    vector={
                                        "text": text_vector
                                    }
                                )
                            ]
                        )
                        count_success += 1
                        if count_success % 50 == 0:
                            print(f"{count_success} Textvektoren aktualisiert...")
                    except Exception as e:
                        print(f"Fehler bei {filename}: {e}")
                        count_error += 1
                else:
                    count_skipped += 1
            else:
                count_skipped += 1
                
        offset = next_offset
        if offset is None:
            break
            
    print(f"Text Update abgeschlossen! Erfolgreich: {count_success}, Übersprungen: {count_skipped}, Fehler: {count_error}")

@router.post("/update-text-all", dependencies=[Depends(verify_index_key)])
async def update_all_text(background_tasks: BackgroundTasks):
    """
    Aktualisiert alle Textvektoren für Bilder mit delta=0 im Hintergrund.
    """
    background_tasks.add_task(process_update_all_text)
    return {"message": "Update aller Textvektoren wurde im Hintergrund gestartet."}

class DeleteByNidRequest(BaseModel):
    nid: int

@router.post("/delete-by-nid", dependencies=[Depends(verify_index_key)])
async def delete_by_nid(request: DeleteByNidRequest):
    """
    Löscht alle Vektoren aus Qdrant, die zu einer bestimmten NID (Person) gehören.
    """
    qdrant = get_qdrant_client()
    if not qdrant:
        raise HTTPException(status_code=500, detail="Qdrant Client nicht verfügbar.")

    try:
        print(f"Lösche alle Vektoren für NID {request.nid} aus Qdrant...")
        
        # Wir löschen alle Punkte, deren NID-Feld in der Payload übereinstimmt
        operation_info = qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="nid",
                        match=MatchValue(value=request.nid)
                    )
                ]
            ),
        )

        return {"message": f"Vektoren für NID {request.nid} erfolgreich gelöscht.", "status": operation_info.status}
    except Exception as e:
        print(f"Fehler beim Löschen für NID {request.nid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteByFilenameRequest(BaseModel):
    filename: str

@router.post("/delete-one", dependencies=[Depends(verify_index_key)])
async def delete_single_image(request: DeleteByFilenameRequest):
    """
    Löscht einen einzelnen Vektor anhand des Dateinamens aus Qdrant.
    """
    key = request.filename
    qdrant = get_qdrant_client()
    if not qdrant:
        raise HTTPException(status_code=500, detail="Qdrant Client nicht verfügbar.")

    try:
        print(f"Lösche Vektor für Bild {key} aus Qdrant...")
        
        # Da wir eine deterministische UUID für den Dateinamen vergeben haben
        # beim Indexieren, können wir sie auch wieder neu berechnen.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
        
        # Alternativ über Payload Filter löschen (sicherer falls der ID-Algorithmus gewechselt wurde)
        operation_info = qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=key)
                    )
                ]
            ),
        )
        
        return {"message": f"Bild '{key}' erfolgreich aus Qdrant gelöscht.", "status": operation_info.status}

    except Exception as e:
        print(f"Fehler beim Löschen vom Bild {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

