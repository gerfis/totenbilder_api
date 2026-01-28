import os
import uuid
import torch
import io
import boto3
from dotenv import load_dotenv
from PIL import Image
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header, Depends, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

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

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

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
        print(f"Lade CLIP IMAGE Model auf {DEVICE}...")
        _model_img = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
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

def process_indexing(force_reindex: bool = False):
    """
    Hintergrund-Funktion, die den R2 Bucket durchläuft und Bilder indexiert.
    """
    s3 = get_s3_client()
    qdrant = get_qdrant_client()
    model = get_model()

    if not s3 or not qdrant or not model:
        print("Fehler: Clients konnten nicht initialisiert werden.")
        return

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
                img = Image.open(io.BytesIO(file_content))
                
                # Embedden
                vector = model.encode(img).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"filename": key}
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
        try:
            file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            file_content = file_obj['Body'].read()
            img = Image.open(io.BytesIO(file_content))
        except Exception as e:
            print(f"Fehler beim Laden von {key}: {e}")
            raise HTTPException(status_code=404, detail=f"Bild '{key}' konnte nicht geladen werden: {e}")
        
        # Embedden
        vector = model.encode(img).tolist()
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"filename": key}
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

@router.post("/index", dependencies=[Depends(verify_index_key)])
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Startet die Indexierung der Bilder vom R2 Bucket in Qdrant im Hintergrund.
    """
    background_tasks.add_task(process_indexing, request.force_reindex)
    return {"message": "Indexierung wurde im Hintergrund gestartet.", "bucket": R2_BUCKET_NAME}
