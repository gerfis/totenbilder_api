import os
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import mysql.connector

load_dotenv()

# --- KONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
COLLECTION_IMAGES = os.getenv("QDRANT_COLLECTION_IMAGES", "totenbilder_v2")
COLLECTION_TEXTS = os.getenv("QDRANT_COLLECTION_TEXTS", "totenbilder_texte")
COLLECTION_GEMINI = os.getenv("QDRANT_COLLECTION_GEMINI", "totenbilder_gemini_768")
PUBLIC_IMAGE_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import google.genai
from google.genai import types as genai_types

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


router = APIRouter()

# Globale Variablen (Modul-Level)
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

def generate_gemini_embedding(text_content: str):
    client = get_gemini_client()
    if not client:
        return None
        
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[text_content],
            config=genai_types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"WARNUNG: Gemini Text Embedding fehlgeschlagen: {e}")
        return None

# CLIP and MiniLM local models removed to save RAM. 
# Using Gemini 2 for all embeddings.

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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

class SearchQuery(BaseModel):
    query: Optional[str] = None
    similar: Optional[str] = None
    limit: int = 30
    offset: int = 0
    delta: Optional[str] = "alle"
    type: str = "image" # "image" (CLIP) oder "text" (MiniLM)
    method: str = "gemini" # Default gemini, others ignored.

class SearchResult(BaseModel):
    filename: str
    image_url: str
    score: float
    nid: Optional[int] = None
    delta: Optional[int] = None
    field_type: Optional[str] = None
    text_content: Optional[str] = None

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

    # Gemini 2 is the exclusive search method.
    # The 'method' parameter is kept for compatibility but ignored.

    if search_req.similar:
        print(f"Suche nach ähnlichen Bildern (Gemini): '{search_req.similar}'")
        ref_points = client.scroll(
            collection_name=COLLECTION_GEMINI,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=search_req.similar))]
            ),
            limit=1,
            with_vectors=True
        )
        
        if ref_points[0]:
            ref_vector = ref_points[0][0].vector
            points = client.query_points(
                collection_name=COLLECTION_GEMINI,
                query=ref_vector,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=search_req.limit,
                offset=search_req.offset
            ).points
            
            for hit in points:
                results.append(create_result(hit))
        else:
             raise HTTPException(status_code=404, detail=f"Bild '{search_req.similar}' in Gemini nicht gefunden.")

    elif search_req.query:
        # Gemini handles both descriptive 'image' search and 'text' search natively.
        # For 'text' search where we want specific matching against OCR'd fields,
        # we still use the Gemini collection since it contains the combined text.
        
        print(f"Suche nach '{search_req.query}' (Gemini)")
        text_vector = generate_gemini_embedding(search_req.query)
        if not text_vector:
             raise HTTPException(status_code=500, detail="Gemini Embedding fehlgeschlagen.")
        
        points = client.query_points(
            collection_name=COLLECTION_GEMINI,
            query=text_vector,
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=search_req.limit * 5 if search_req.type == "text" else search_req.limit,
            offset=search_req.offset
        ).points
        
        seen_identifiers = set()
        for hit in points:
            # Deduplicate by nid, fallback to filename
            identifier = hit.payload.get("nid") or hit.payload.get("filename")
            if identifier and identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                results.append(create_result(hit))
                if len(results) >= search_req.limit:
                    break
    else:
        return []

    return results

@router.get("/search", response_model=List[SearchResult])
async def search_images_get(query: str, limit: int = 30, offset: int = 0, delta: str = "alle", type: str = "image", method: str = "fastembed"):
    """
    Ermöglicht die Text-Suche per GET-Request.
    """
    return await search_images(SearchQuery(query=query, limit=limit, offset=offset, delta=delta, type=type, method=method))

def create_result(hit):
    fname_key = hit.payload.get("filename")
    nid = hit.payload.get("nid")
    delta = hit.payload.get("delta")
    field_type = hit.payload.get("field_type")
    text_content = hit.payload.get("text_content")
    score = round(hit.score, 3)
    
    # URL Konstruktion
    base = PUBLIC_IMAGE_BASE_URL.rstrip("/") if PUBLIC_IMAGE_BASE_URL else ""
    image_url = f"{base}/{fname_key}"
    
    return SearchResult(
        filename=fname_key,
        image_url=image_url,
        score=score,
        nid=nid,
        delta=delta,
        field_type=field_type,
        text_content=text_content
    )

class LatestResult(BaseModel):
    nid: int
    Name: str
    Sterbedatum: Optional[str] = None
    Sterbetag: Optional[int] = None
    Sterbemonat: Optional[int] = None
    Sterbejahr: Optional[int] = None
    Wohnort: Optional[str] = None
    Ort: Optional[str] = None
    url: str
    alias: Optional[str] = None

@router.get("/latest", response_model=List[LatestResult])
async def get_latest(anzahl: int = 10, wohnort: Optional[str] = None, ort: Optional[str] = None):
    """
    Gibt die letzten <anzahl> Totenbilder als JSON aus, nach nid absteigend sortiert.
    Optional kann nach Wohnort (oder Ort) gefiltert werden.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        cursor = conn.cursor(dictionary=True)
        search_ort = wohnort or ort
        if search_ort:
            query = """
                SELECT t.nid, t.Name, t.Sterbedatum, t.Sterbetag, t.Sterbemonat, t.Sterbejahr, t.Wohnort, t.alias, b.filename
                FROM totenbilder t
                JOIN totenbilder_bilder b ON t.nid = b.nid
                WHERE b.delta = 0 AND t.Wohnort = %s
                ORDER BY t.Sterbedatum DESC
                LIMIT %s
            """
            cursor.execute(query, (search_ort, anzahl))
        else:
            query = """
                SELECT t.nid, t.Name, t.Sterbedatum, t.Sterbetag, t.Sterbemonat, t.Sterbejahr, t.Wohnort, t.alias, b.filename
                FROM totenbilder t
                JOIN totenbilder_bilder b ON t.nid = b.nid
                WHERE b.delta = 0
                ORDER BY t.Sterbedatum DESC
                LIMIT %s
            """
            cursor.execute(query, (anzahl,))
            
        rows = cursor.fetchall()
        
        results = []
        base = PUBLIC_IMAGE_BASE_URL.rstrip("/") if PUBLIC_IMAGE_BASE_URL else ""
        r2_prefix = os.getenv("R2_PREFIX", "totenbilder/")
        
        for row in rows:
            fname_key = row["filename"]
            # Sicherstellen, dass der Prefix vorhanden ist, falls noetig
            if not fname_key.startswith(r2_prefix):
                fname_key = f"{r2_prefix}{fname_key}"
            
            image_url = f"{base}/{fname_key}"
            
            results.append({
                "nid": row["nid"],
                "Name": row["Name"] or "",
                "Sterbedatum": row["Sterbedatum"],
                "Sterbetag": row["Sterbetag"],
                "Sterbemonat": row["Sterbemonat"],
                "Sterbejahr": row["Sterbejahr"],
                "Wohnort": row["Wohnort"],
                "Ort": row["Wohnort"],
                "url": image_url,
                "alias": row["alias"]
            })
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.get("/today", response_model=List[LatestResult])
async def get_today(anzahl: Optional[int] = None, wohnort: Optional[str] = None, ort: Optional[str] = None, tag: Optional[int] = None, monat: Optional[int] = None):
    """
    Gibt Totenbilder aus, deren Sterbetag und -monat dem heutigen Tag (oder den explizit übergebenen Parametern) entsprechen.
    Wenn anzahl nicht gesetzt ist (Standard), werden alle passenden Bilder ausgegeben.
    Ist heute der 29. Februar (Schaltjahr) und keine expliziten Parameter wurden gesetzt, werden nur Bilder mit Sterbetag 29.2. angezeigt.
    """
    if tag is not None and monat is not None:
        target_day = tag
        target_month = monat
    else:
        now = datetime.now()
        target_day = now.day
        target_month = now.month

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    try:
        cursor = conn.cursor(dictionary=True)
        query_params = [target_day, target_month]
        
        base_query = """
            SELECT t.nid, t.Name, t.Sterbedatum, t.Sterbetag, t.Sterbemonat, t.Sterbejahr, t.Wohnort, t.alias, b.filename
            FROM totenbilder t
            JOIN totenbilder_bilder b ON t.nid = b.nid
            WHERE b.delta = 0 AND t.Sterbetag = %s AND t.Sterbemonat = %s
        """
        
        search_ort = wohnort or ort
        if search_ort:
            base_query += " AND t.Wohnort = %s"
            query_params.append(search_ort)
            
        base_query += " ORDER BY t.Sterbedatum DESC"
        
        if anzahl is not None:
            base_query += " LIMIT %s"
            query_params.append(anzahl)
            
        cursor.execute(base_query, tuple(query_params))
        rows = cursor.fetchall()
        
        results = []
        base = PUBLIC_IMAGE_BASE_URL.rstrip("/") if PUBLIC_IMAGE_BASE_URL else ""
        r2_prefix = os.getenv("R2_PREFIX", "totenbilder/")
        
        for row in rows:
            fname_key = row["filename"]
            if not fname_key.startswith(r2_prefix):
                fname_key = f"{r2_prefix}{fname_key}"
            
            image_url = f"{base}/{fname_key}"
            
            results.append({
                "nid": row["nid"],
                "Name": row["Name"] or "",
                "Sterbedatum": row["Sterbedatum"],
                "Sterbetag": row["Sterbetag"],
                "Sterbemonat": row["Sterbemonat"],
                "Sterbejahr": row["Sterbejahr"],
                "Wohnort": row["Wohnort"],
                "Ort": row["Wohnort"],
                "url": image_url,
                "alias": row["alias"]
            })
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
