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
import google.genai
from google.genai import types as genai_types

load_dotenv()

# --- KONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PREFIX = os.getenv("R2_PREFIX")
COLLECTION_IMAGES = os.getenv("QDRANT_COLLECTION_IMAGES", "totenbilder_v2")
COLLECTION_TEXTS = os.getenv("QDRANT_COLLECTION_TEXTS", "totenbilder_texte")
COLLECTION_GEMINI = os.getenv("QDRANT_COLLECTION_GEMINI", "totenbilder_gemini_768")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if GEMINI_API_KEY:
            _gemini_client = google.genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={'headers': {'referer': 'https://admin.totenbilder.at'}}
            )
        else:
            print("WARNUNG: GEMINI_API_KEY nicht gesetzt!")
    return _gemini_client

def generate_gemini_embedding(file_content: bytes, filename: str, combined_text: str = ""):
    client = get_gemini_client()
    if not client:
        return None
        
    mime_type = "image/jpeg"
    if filename.lower().endswith('.png'):
        mime_type = "image/png"
    elif filename.lower().endswith('.webp'):
        mime_type = "image/webp"

    contents = [
        genai_types.Part.from_bytes(data=file_content, mime_type=mime_type)
    ]
    if combined_text:
        contents.append(combined_text)
        
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=contents,
            config=genai_types.EmbedContentConfig(output_dimensionality=768)
        )
        # response is likely EmbedContentResponse, which has a list of embeddings
        return response.embeddings[0].values
    except Exception as e:
        print(f"WARNUNG: Gemini Embedding fehlgeschlagen für {filename}: {e}")
        return None

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
        
        # Init Images Collection
        if not _qdrant_client.collection_exists(COLLECTION_IMAGES):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_IMAGES,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        # Init Texts Collection
        if not _qdrant_client.collection_exists(COLLECTION_TEXTS):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_TEXTS,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
        # Init Gemini Collection
        if not _qdrant_client.collection_exists(COLLECTION_GEMINI):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_GEMINI,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            
        # Payload Indices sicherstellen
        try:
            for coll in [COLLECTION_IMAGES, COLLECTION_TEXTS, COLLECTION_GEMINI]:
                _qdrant_client.create_payload_index(collection_name=coll, field_name="filename", field_schema="keyword")
                _qdrant_client.create_payload_index(collection_name=coll, field_name="nid", field_schema="integer")
                _qdrant_client.create_payload_index(collection_name=coll, field_name="delta", field_schema="integer")
            # Text collection needs field_type index too
            _qdrant_client.create_payload_index(collection_name=COLLECTION_TEXTS, field_name="field_type", field_schema="keyword")
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
    Fetches all NID, Delta and categorized fields from MySQL and returns a dict:
    { "filename": {"nid": 123, "delta": 0, "fields": {"Name": "Hans", ...} }, ... }
    """
    print("Fetching metadata from MySQL...")
    conn = get_db_connection()
    if not conn:
        return {}
    
    metadata_map = {}
    try:
        cursor = conn.cursor(dictionary=True)
        # JOIN table "totenbilder" to get individual fields
        query = """
        SELECT b.filename, b.nid, b.delta, 
               t.Name, t.Nachname, t.Ledigname, t.Ort, t.Strasse, t.Begraebnisort, 
               t.Beruf1, t.Beruf2, t.Ehrenaemter, t.Bemerkung, t.Trauerspruch, 
               t.Bildinhalt, t.Todesgrund
        FROM totenbilder_bilder b 
        LEFT JOIN totenbilder t ON b.nid = t.nid
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        fields_to_check = ['Name', 'Nachname', 'Ledigname', 'Ort', 'Strasse', 
                           'Begraebnisort', 'Beruf1', 'Beruf2', 'Ehrenaemter', 
                           'Bemerkung', 'Trauerspruch', 'Bildinhalt', 'Todesgrund']
                           
        for row in results:
            fields_data = {}
            for f in fields_to_check:
                if row.get(f) and str(row.get(f)).strip():
                    fields_data[f] = str(row[f]).strip()
                    
            metadata_map[row['filename']] = {
                "nid": row['nid'], 
                "delta": row['delta'],
                "fields": fields_data
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

    # 1. Collections neu erstellen?
    if recreate_collection:
        print(f"!!! ACHTUNG: Lösche und erstelle Collections neu...")
        qdrant.recreate_collection(
            collection_name=COLLECTION_IMAGES,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        qdrant.recreate_collection(
            collection_name=COLLECTION_TEXTS,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        qdrant.recreate_collection(
            collection_name=COLLECTION_GEMINI,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        # Payload Index sofort wieder anlegen
        for coll in [COLLECTION_IMAGES, COLLECTION_TEXTS, COLLECTION_GEMINI]:
            qdrant.create_payload_index(collection_name=coll, field_name="filename", field_schema="keyword")
            qdrant.create_payload_index(collection_name=coll, field_name="nid", field_schema="integer")
            qdrant.create_payload_index(collection_name=coll, field_name="delta", field_schema="integer")
        qdrant.create_payload_index(collection_name=COLLECTION_TEXTS, field_name="field_type", field_schema="keyword")
        force_reindex = True # Logischerweise müssen wir dann alles neu machen

    print(f"--- Starte Indexierung. Bucket: {R2_BUCKET_NAME} (Ordner: {R2_PREFIX}) ---")
    image_points_buffer = []
    text_points_buffer = []
    gemini_points_buffer = []
    
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
                        collection_name=COLLECTION_IMAGES,
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
                img_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

                # Payload zusammenbauen
                img_payload = {"filename": key}
                db_filename = key.replace(R2_PREFIX, "")
                
                # Calculate combined text for Gemini
                combined_text = ""
                # Metadata and Chunking Text Fields
                if db_filename in metadata_map:
                    img_payload["nid"] = metadata_map[db_filename]["nid"]
                    img_payload["delta"] = metadata_map[db_filename]["delta"]
                    
                    if img_payload["delta"] == 0 and metadata_map[db_filename].get("fields"):
                        combined_text = " ".join(metadata_map[db_filename]["fields"].values())
                        for field_name, field_value in metadata_map[db_filename]["fields"].items():
                            try:
                                text_vector = list(model_txt.embed([field_value]))[0].tolist()
                                # Deterministic ID based on filename AND field_name
                                text_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}_{field_name}"))
                                text_points_buffer.append(PointStruct(
                                    id=text_point_id,
                                    vector=text_vector,
                                    payload={
                                        "filename": key,
                                        "nid": img_payload["nid"],
                                        "delta": img_payload["delta"],
                                        "field_type": field_name,
                                        "text_content": field_value
                                    }
                                ))
                            except Exception as text_e:
                                print(f"WARNUNG: Text Embedding fehlgeschlagen für {db_filename} ({field_name}): {text_e}")
                
                # Create Image Point
                img_point = PointStruct(
                    id=img_point_id,
                    vector=img_vector,
                    payload=img_payload
                )
                image_points_buffer.append(img_point)
                
                # Create Gemini Point
                gemini_vector = generate_gemini_embedding(file_content, key, combined_text)
                if gemini_vector:
                    gemini_point = PointStruct(
                        id=img_point_id, # Can share the same UUID based on key
                        vector=gemini_vector,
                        payload=img_payload
                    )
                    gemini_points_buffer.append(gemini_point)
                
                count_processed += 1
                
                # Batch-Upload
                if len(image_points_buffer) >= 50:
                    qdrant.upsert(collection_name=COLLECTION_IMAGES, points=image_points_buffer)
                    image_points_buffer = []
                    print(f"Fortschritt: {count_processed} Bilder neu verarbeitet...")
                if len(gemini_points_buffer) >= 50:
                    qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
                    gemini_points_buffer = []
                if len(text_points_buffer) >= 150: # Text buffer can grow faster
                    qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points_buffer)
                    text_points_buffer = []
                    
            except Exception as e:
                print(f"Fehler bei {key}: {e}")

    if image_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_IMAGES, points=image_points_buffer)
    if text_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points_buffer)
    if gemini_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
    
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
        text_points = []
        combined_text = ""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                db_filename_stripped = key.replace(R2_PREFIX, "")
                cursor.execute(
                    """
                    SELECT b.nid, b.delta, 
                           t.Name, t.Nachname, t.Ledigname, t.Ort, t.Strasse, t.Begraebnisort, 
                           t.Beruf1, t.Beruf2, t.Ehrenaemter, t.Bemerkung, t.Trauerspruch, 
                           t.Bildinhalt, t.Todesgrund
                    FROM totenbilder_bilder b 
                    LEFT JOIN totenbilder t ON b.nid=t.nid 
                    WHERE b.filename = %s OR b.filename = %s
                    """, 
                    (db_filename_stripped, key)
                )
                row = cursor.fetchone()
                if row:
                    payload["nid"] = row['nid']
                    payload["delta"] = row['delta']
                    
                    if payload["delta"] == 0:
                        fields_to_check = ['Name', 'Nachname', 'Ledigname', 'Ort', 'Strasse', 
                                         'Begraebnisort', 'Beruf1', 'Beruf2', 'Ehrenaemter', 
                                         'Bemerkung', 'Trauerspruch', 'Bildinhalt', 'Todesgrund']
                        texts = []
                        for f in fields_to_check:
                            if row.get(f) and str(row.get(f)).strip():
                                texts.append(str(row[f]).strip())
                        combined_text = " ".join(texts)
                else:
                    print(f"WARNUNG: Keine Metadaten für {key} gefunden!")
                    
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error fetching metadata for single image: {e}")

        img_point = PointStruct(
            id=point_id,
            vector=img_vector,
            payload=payload
        )
        
        # Upsert (direkt, ohne Buffer)
        qdrant.upsert(collection_name=COLLECTION_IMAGES, points=[img_point])
        if text_points:
            qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points)
            
        gemini_vector = generate_gemini_embedding(file_content, key, combined_text)
        if gemini_vector:
            gemini_point = PointStruct(
                id=point_id,
                vector=gemini_vector,
                payload=payload
            )
            qdrant.upsert(collection_name=COLLECTION_GEMINI, points=[gemini_point])
        
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
        print(f"Update Textvektoren für: {key}...")
        
        # Metadata fetch
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="DB Error")
            
        cursor = conn.cursor(dictionary=True)
        db_filename_stripped = key.replace(R2_PREFIX, "")
        cursor.execute(
            """
            SELECT b.nid, b.delta, 
                   t.Name, t.Nachname, t.Ledigname, t.Ort, t.Strasse, t.Begraebnisort, 
                   t.Beruf1, t.Beruf2, t.Ehrenaemter, t.Bemerkung, t.Trauerspruch, 
                   t.Bildinhalt, t.Todesgrund
            FROM totenbilder_bilder b 
            LEFT JOIN totenbilder t ON b.nid=t.nid 
            WHERE b.filename = %s OR b.filename = %s
            """, 
            (db_filename_stripped, key)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Eintrag in DB nicht gefunden.")
            
        if row['delta'] != 0:
            return {"message": f"Übersprungen: {key} ist eine Rückseite (delta>0).", "filename": key}
            
        # Zuerst alte Textvektoren für dieses Bild löschen
        qdrant.delete(
            collection_name=COLLECTION_TEXTS,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=key))]
            ),
        )
        
        nid = row['nid']
        delta = row['delta']
        
        text_points = []
        fields_to_check = ['Name', 'Nachname', 'Ledigname', 'Ort', 'Strasse', 
                           'Begraebnisort', 'Beruf1', 'Beruf2', 'Ehrenaemter', 
                           'Bemerkung', 'Trauerspruch', 'Bildinhalt', 'Todesgrund']
        
        for f in fields_to_check:
            if row.get(f) and str(row.get(f)).strip():
                field_value = str(row[f]).strip()
                try:
                    text_vector = list(model_txt.embed([field_value]))[0].tolist()
                    text_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}_{f}"))
                    text_points.append(PointStruct(
                        id=text_point_id,
                        vector=text_vector,
                        payload={
                            "filename": key,
                            "nid": nid,
                            "delta": delta,
                            "field_type": f,
                            "text_content": field_value
                        }
                    ))
                except Exception as text_e:
                    print(f"WARNUNG: Text Embedding fehlgeschlagen für {key} ({f}): {text_e}")

        if text_points:
            qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points)
        
        return {"message": f"Textvektoren für '{key}' erfolgreich aktualisiert.", "filename": key}

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
    
    print(f"!!! ACHTUNG: Lösche und erstelle Collection '{COLLECTION_TEXTS}' neu...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_TEXTS,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    for coll in [COLLECTION_TEXTS]:
        qdrant.create_payload_index(collection_name=coll, field_name="filename", field_schema="keyword")
        qdrant.create_payload_index(collection_name=coll, field_name="nid", field_schema="integer")
        qdrant.create_payload_index(collection_name=coll, field_name="delta", field_schema="integer")
        qdrant.create_payload_index(collection_name=coll, field_name="field_type", field_schema="keyword")

    text_points_buffer = []
    count_success = 0
    
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_IMAGES,
            with_payload=["filename", "delta", "nid"],
            with_vectors=False,
            limit=500,
            offset=offset
        )
        
        for point in points:
            filename = point.payload.get("filename")
            delta = point.payload.get("delta", 0)
            
            if not filename or delta != 0:
                continue
                
            db_filename = filename.replace(R2_PREFIX, "")
            
            if db_filename in metadata_map and metadata_map[db_filename].get("fields"):
                for field_name, field_value in metadata_map[db_filename]["fields"].items():
                    try:
                        text_vector = list(model_txt.embed([field_value]))[0].tolist()
                        text_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{field_name}"))
                        text_points_buffer.append(PointStruct(
                            id=text_point_id,
                            vector=text_vector,
                            payload={
                                "filename": filename,
                                "nid": point.payload.get("nid"),
                                "delta": delta,
                                "field_type": field_name,
                                "text_content": field_value
                            }
                        ))
                    except Exception as e:
                        print(f"Fehler bei {filename} ({field_name}): {e}")
                
            if len(text_points_buffer) >= 150:
                qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points_buffer)
                count_success += len(text_points_buffer)
                print(f"Fortschritt: {count_success} Text-Chunks generiert...")
                text_points_buffer = []

        offset = next_offset
        if offset is None:
            break
            
    if text_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_TEXTS, points=text_points_buffer)
        count_success += len(text_points_buffer)
            
    print(f"Text Update abgeschlossen! Erfolgreich angelegt: {count_success} Text-Chunks.")

@router.post("/update-text-all", dependencies=[Depends(verify_index_key)])
async def update_all_text(background_tasks: BackgroundTasks):
    """
    Aktualisiert alle Textvektoren für Bilder mit delta=0 im Hintergrund.
    """
    background_tasks.add_task(process_update_all_text)
    return {"message": "Update aller Textvektoren wurde im Hintergrund gestartet."}

def process_update_all_gemini():
    """Hintergrund-Task zum Durchlaufen aller Bilder und Neu-Erstellen der Gemini Embeddings"""
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    
    if not s3 or not qdrant:
        print("Fehler: Clients nicht verfügbar")
        return
        
    print(f"!!! ACHTUNG: Lösche und erstelle Collection '{COLLECTION_GEMINI}' neu...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_GEMINI,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    for coll in [COLLECTION_GEMINI]:
        qdrant.create_payload_index(collection_name=coll, field_name="filename", field_schema="keyword")
        qdrant.create_payload_index(collection_name=coll, field_name="nid", field_schema="integer")
        qdrant.create_payload_index(collection_name=coll, field_name="delta", field_schema="integer")
        
    print("Fetching metadata for Gemini...")
    metadata_map = fetch_all_metadata()
    
    gemini_points_buffer = []
    count_success = 0
    count_skipped = 0
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=R2_PREFIX)
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key == R2_PREFIX or not key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
                
            try:
                db_filename = key.replace(R2_PREFIX, "")
                combined_text = ""
                payload = {"filename": key}
                
                if db_filename in metadata_map:
                    payload["nid"] = metadata_map[db_filename]["nid"]
                    payload["delta"] = metadata_map[db_filename]["delta"]
                    
                    if payload["delta"] == 0 and metadata_map[db_filename].get("fields"):
                        combined_text = " ".join(metadata_map[db_filename]["fields"].values())
                
                file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
                file_content = file_obj['Body'].read()
                
                gemini_vector = generate_gemini_embedding(file_content, key, combined_text)
                if gemini_vector:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
                    gemini_point = PointStruct(
                        id=point_id,
                        vector=gemini_vector,
                        payload=payload
                    )
                    gemini_points_buffer.append(gemini_point)
                    count_success += 1
                else:
                    count_skipped += 1
                    
                if len(gemini_points_buffer) >= 50:
                    qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
                    gemini_points_buffer = []
                    print(f"Fortschritt: {count_success} Gemini Embeddings generiert...")
                    
            except Exception as e:
                print(f"Fehler bei {key}: {e}")
                count_skipped += 1

    if gemini_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
        
    print(f"Gemini Update abgeschlossen! Erfolgreich: {count_success}, Übersprungen/Fehler: {count_skipped}")

def process_gemini_test_index():
    """Hintergrund-Task für Test-Indexierung der 100 neuesten Bilder für Gemini"""
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    
    if not s3 or not qdrant:
        print("Fehler: Clients nicht verfügbar")
        return
        
    # Optional: Delete and recreate collection for a clean test? The user implied it should only be done for full index.
    # But to make sure it exists:
    if not qdrant.collection_exists(COLLECTION_GEMINI):
        qdrant.create_collection(
            collection_name=COLLECTION_GEMINI,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        for coll in [COLLECTION_GEMINI]:
            qdrant.create_payload_index(collection_name=coll, field_name="filename", field_schema="keyword")
            qdrant.create_payload_index(collection_name=coll, field_name="nid", field_schema="integer")
            qdrant.create_payload_index(collection_name=coll, field_name="delta", field_schema="integer")

    print("Fetching metadata for Gemini (latest 100)...")
    
    conn = get_db_connection()
    if not conn:
        print("Datenbank nicht erreichbar!")
        return

    latest_images = []
    try:
        cursor = conn.cursor(dictionary=True)
        # Neueste 100 Bilder aus der Datenbank holen
        query = """
        SELECT b.filename, b.nid, b.delta, 
               t.Name, t.Nachname, t.Ledigname, t.Ort, t.Strasse, t.Begraebnisort, 
               t.Beruf1, t.Beruf2, t.Ehrenaemter, t.Bemerkung, t.Trauerspruch, 
               t.Bildinhalt, t.Todesgrund
        FROM totenbilder_bilder b 
        LEFT JOIN totenbilder t ON b.nid = t.nid
        ORDER BY b.nid DESC
        LIMIT 100
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        fields_to_check = ['Name', 'Nachname', 'Ledigname', 'Ort', 'Strasse', 
                           'Begraebnisort', 'Beruf1', 'Beruf2', 'Ehrenaemter', 
                           'Bemerkung', 'Trauerspruch', 'Bildinhalt', 'Todesgrund']
                           
        for row in results:
            fields_data = {}
            for f in fields_to_check:
                if row.get(f) and str(row.get(f)).strip():
                    fields_data[f] = str(row[f]).strip()
            
            # Reconstruct the S3 prefix key
            s3_key = R2_PREFIX + row['filename']
            latest_images.append({
                "key": s3_key,
                "nid": row['nid'], 
                "delta": row['delta'],
                "fields": fields_data
            })
            
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    gemini_points_buffer = []
    count_success = 0
    count_skipped = 0
    
    for item in latest_images:
        key = item["key"]
        
        if not key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
            
        try:
            combined_text = ""
            payload = {
                "filename": key,
                "nid": item["nid"],
                "delta": item["delta"]
            }
            
            if payload["delta"] == 0 and item.get("fields"):
                combined_text = " ".join(item["fields"].values())
            
            # Bild aus R2 laden
            file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            file_content = file_obj['Body'].read()
            
            gemini_vector = generate_gemini_embedding(file_content, key, combined_text)
            if gemini_vector:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
                gemini_point = PointStruct(
                    id=point_id,
                    vector=gemini_vector,
                    payload=payload
                )
                gemini_points_buffer.append(gemini_point)
                count_success += 1
            else:
                count_skipped += 1
                
            if len(gemini_points_buffer) >= 50:
                qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
                print(f"Fortschritt: {count_success} von {len(latest_images)} Gemini Embeddings generiert...")
                gemini_points_buffer = []
                
        except Exception as e:
            print(f"Fehler bei {key}: {e}")
            count_skipped += 1

    if gemini_points_buffer:
        qdrant.upsert(collection_name=COLLECTION_GEMINI, points=gemini_points_buffer)
        
    print(f"Gemini Test Update abgeschlossen! Erfolgreich: {count_success}, Übersprungen/Fehler: {count_skipped}")

@router.post("/index-gemini-test", dependencies=[Depends(verify_index_key)])
async def trigger_index_gemini_test(background_tasks: BackgroundTasks):
    """
    Startet die Test-Indexierung der neusten 100 Bilder für Gemini im Hintergrund.
    """
    background_tasks.add_task(process_gemini_test_index)
    return {"message": "Gemini-Test-Indexierung (100 neueste Bilder) wurde im Hintergrund gestartet."}

@router.post("/index-all-gemini", dependencies=[Depends(verify_index_key)])
async def trigger_index_all_gemini(background_tasks: BackgroundTasks):
    """
    Startet die Indexierung der Bilder exklusiv für Gemini im Hintergrund.
    Dies löscht die Gemini Collection vorher.
    """
    background_tasks.add_task(process_update_all_gemini)
    return {"message": "Gemini-Indexierung wurde im Hintergrund gestartet."}

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
        
        # Wir löschen alle Punkte, deren NID-Feld in der Payload übereinstimmt, in beiden Collections
        status_info = []
        for coll in [COLLECTION_IMAGES, COLLECTION_TEXTS, COLLECTION_GEMINI]:
            operation_info = qdrant.delete(
                collection_name=coll,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="nid",
                            match=MatchValue(value=request.nid)
                        )
                    ]
                ),
            )
            status_info.append(operation_info.status)

        return {"message": f"Vektoren für NID {request.nid} erfolgreich gelöscht.", "status": str(status_info)}
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
        
        status_info = []
        for coll in [COLLECTION_IMAGES, COLLECTION_TEXTS, COLLECTION_GEMINI]:
            operation_info = qdrant.delete(
                collection_name=coll,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=key)
                        )
                    ]
                ),
            )
            status_info.append(operation_info.status)
        
        return {"message": f"Bild '{key}' erfolgreich aus Qdrant gelöscht.", "status": str(status_info)}

    except Exception as e:
        print(f"Fehler beim Löschen vom Bild {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

