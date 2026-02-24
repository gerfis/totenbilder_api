import os
import urllib.parse
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from deep_translator import GoogleTranslator

load_dotenv()

# --- KONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
PUBLIC_IMAGE_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL")



router = APIRouter()

# Globale Variablen (Modul-Level)
_model_text = None
_model_minilm = None
_qdrant_client = None

def get_model():
    global _model_text
    if _model_text is None:
        print(f"Lade CLIP TEXT Model (fastembed)...")
        _model_text = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    return _model_text

def get_minilm_model():
    global _model_minilm
    if _model_minilm is None:
        print(f"Lade MiniLM TEXT Model (fastembed)...")
        _model_minilm = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _model_minilm

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client

class SearchQuery(BaseModel):
    query: Optional[str] = None
    similar: Optional[str] = None
    limit: int = 30
    offset: int = 0
    delta: Optional[str] = "alle"
    type: str = "image" # "image" (CLIP) oder "text" (MiniLM)

class SearchResult(BaseModel):
    filename: str
    image_url: str
    score: float
    nid: Optional[int] = None
    delta: Optional[int] = None

@router.post("/search", response_model=List[SearchResult])
async def search_images(search_req: SearchQuery):
    client = get_qdrant_client()
    results = []

    # Filter vorbereiten
    filter_conditions = []
    # delta Filterlogik
    # Wir unterstützen: "alle" (kein Filter), "0" (delta==0), ">0" (delta > 0)
    # Default ist "alle" via Model


    if search_req.delta == "0":
         # Exakt 0
         filter_conditions.append(FieldCondition(key="delta", match=MatchValue(value=0)))
    elif search_req.delta == ">0":
         # Größer 0
         filter_conditions.append(FieldCondition(key="delta", range=Range(gt=0)))

    if search_req.similar:

        
        # 1. Vektor des Referenzbildes holen
        # Anm.: Wir filtern NICHT beim Holen des Referenzbildes, sondern beim Suchen der Ähnlichen.
        # Das Referenzbild selbst suchen wir nur per filename.
        ref_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=search_req.similar))]
            ),

            limit=1,
            with_vectors=True
        )
        
        if ref_points[0]:
            ref_vector = ref_points[0][0].vector
            
            # 2. Ähnliche suchen
            points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=ref_vector,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=search_req.limit,

                offset=search_req.offset
            ).points
            
            for hit in points:
                results.append(create_result(hit))
        else:
             raise HTTPException(status_code=404, detail=f"Bild '{search_req.similar}' nicht gefunden.")

    elif search_req.query:
        if search_req.type == "text":
            # REINE TEXT-SUCHE (über MiniLM)
            print(f"Suche nach Text (MiniLM): '{search_req.query}'")
            local_model = get_minilm_model()
            text_vector = list(local_model.embed([search_req.query]))[0].tolist()
            
            points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_vector,
                using="text",
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=search_req.limit,
                offset=search_req.offset
            ).points
            
            for hit in points:
                results.append(create_result(hit))
                
        else:
            # TEXT-ZU-BILD SUCHE (über CLIP)
            # Übersetzung von Deutsch nach Englisch für bessere Suchergebnisse in CLIP
            try:
                translated_query = GoogleTranslator(source='auto', target='en').translate(search_req.query)
                print(f"Suche nach Bildinhalten (CLIP): '{search_req.query}' -> Übersetzung: '{translated_query}'")
            except Exception as e:
                print(f"Übersetzungsfehler: {e}")
                translated_query = search_req.query
    
            local_model = get_model()
            # FastEmbed returns a generator of numpy arrays
            text_vector = list(local_model.embed([translated_query]))[0].tolist()
            
            # Bei CLIP greifen wir auf den unbenannten Vektor zu (default)
            points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_vector,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=search_req.limit,
                offset=search_req.offset
            ).points
            
            for hit in points:
                results.append(create_result(hit))
    else:
        # Leere Suche = Fehler oder leere Liste? Hier leere Liste.
        return []

    return results

@router.get("/search", response_model=List[SearchResult])
async def search_images_get(query: str, limit: int = 30, offset: int = 0, delta: str = "alle", type: str = "image"):
    """
    Ermöglicht die Text-Suche per GET-Request.
    """
    return await search_images(SearchQuery(query=query, limit=limit, offset=offset, delta=delta, type=type))

def create_result(hit):
    fname_key = hit.payload.get("filename")
    nid = hit.payload.get("nid")
    delta = hit.payload.get("delta")
    score = round(hit.score, 3)
    
    # URL Konstruktion
    base = PUBLIC_IMAGE_BASE_URL.rstrip("/") if PUBLIC_IMAGE_BASE_URL else ""
    image_url = f"{base}/{fname_key}"
    
    return SearchResult(
        filename=fname_key,
        image_url=image_url,
        score=score,
        nid=nid,
        delta=delta
    )
