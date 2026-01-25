import io
import os
import cv2
import numpy as np
import pytesseract
import traceback
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- KONFIGURATION ---
# Handle Qdrant URL robustly - force port 443 if missing for this host
_default_qdrant_url = "https://qdrant.happyhati.com:443"
QDRANT_URL = os.getenv("QDRANT_URL", _default_qdrant_url)
if "qdrant.happyhati.com" in QDRANT_URL and not QDRANT_URL.endswith(":443"):
    print(f"WARNUNG: QDRANT_URL '{QDRANT_URL}' hat keinen Port. Hänge :443 an.")
    QDRANT_URL = f"{QDRANT_URL}:443"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "9E6B2gnZefnd3m5lGCzLkDIJ2PHNs8WG")
COLLECTION_NAME = "totenbilder"

# Globale Variablen (werden beim Start befüllt)
models = {}
qdrant = None

import asyncio

# --- LIFESPAN (Startup & Shutdown Logik) ---

def sync_initialize():
    """
    Diese Funktion führt die blockierenden Aufrufe synchron aus.
    Sie wird in einem eigenen Thread aufgerufen.
    """
    global qdrant
    
    # 1. Verbindung zu Qdrant
    try:
        print(f"HINTERGRUND: Versuche Verbindung zu Qdrant unter: {QDRANT_URL}")
        # Erst lokal erstellen, um zu testen bevor wir die globale Variable setzen
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Testen der Verbindung (blockierend)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"HINTERGRUND: Erfolgreich mit Qdrant verbunden. Collections: {col_names}")

        # Collection erstellen, falls sie noch nicht existiert
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"HINTERGRUND: Collection '{COLLECTION_NAME}' neu erstellt.")
        
        # Erst wenn alles okay ist, global setzen
        qdrant = client
        print(f"HINTERGRUND: Qdrant-Client ist jetzt einsatzbereit.")

    except Exception as e:
        print(f"HINTERGRUND-FEHLER BEI QDRANT: {type(e).__name__}: {str(e)}")
        qdrant = None

    # 2. KI-Modelle laden
    try:
        print("HINTERGRUND: Lade Vision-Modell (für Bilder) 'clip-ViT-B-32'...")
        models["vision"] = SentenceTransformer('clip-ViT-B-32')
        print("HINTERGRUND: Vision-Modell geladen.")

        print("HINTERGRUND: Lade Text-Modell (für Suche) 'clip-ViT-B-32-multilingual-v1'...")
        models["text"] = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        print("HINTERGRUND: Text-Modell geladen.")
        
    except Exception as e:
        print(f"HINTERGRUND-FEHLER BEIM KI-MODELL: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

async def start_background_init():
    """
    Wrapper um die synchrone Initialisierung in einem Thread-Pool auszuführen.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sync_initialize)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan. Startet die Initialisierung, gibt aber SOFORT die Kontrolle zurück.
    """
    # Wir starten den Task und warten NICHT darauf
    asyncio.create_task(start_background_init())
    
    print("API-Server meldet Startbereitschaft. KI lädt im Hintergrund...")
    yield
    print("Server wird beendet.")

# App Initialisierung
app = FastAPI(lifespan=lifespan)

# --- STATISCHE DATEIEN (Frontend) ---
# Macht den Ordner "static" verfügbar
app.mount("/static", StaticFiles(directory="static"), name="static")

# Leitet den Hauptaufruf (/) direkt auf deine index.html
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/suche")
async def read_search():
    return FileResponse('static/search.html')

@app.get("/health")
async def health_check():
    """
    Einfacher Health-Check um zu sehen, ob der Server überhaupt läuft.
    """
    status = {
        "status": "online",
        "qdrant_connected": qdrant is not None,
        "status": "online",
        "qdrant_connected": qdrant is not None,
        "models_loaded": list(models.keys())
    }
    return status

# --- HILFSFUNKTIONEN ---

def preprocess_image_for_ocr(image_bytes):
    """
    Bereitet das Bild für Tesseract vor:
    1. Konvertierung in Graustufen
    2. Binarisierung (Schwarz/Weiß) mittels Otsu-Methode
    """
    # Bytes in ein Numpy-Array umwandeln (für OpenCV)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # User Request: Keine Vorverarbeitung (kein Grayscale, kein Binarisieren),
    # da dies bei Text auf Hintergrund kontraproduktiv war.
    # Wir geben das Bild direkt an Tesseract weiter.
    
    return img

# --- API ENDPOINTS ---

@app.post("/process-totenbild/")
async def process_totenbild(file: UploadFile = File(...), mysql_id: int = 0):
    """
    Hauptfunktion: 
    1. Bild empfangen
    2. OCR mit Fraktur-Unterstützung
    3. Vektorisierung (Embedding)
    4. Speichern in Qdrant
    """
    try:
        # --- Prüfen, ob Modell und Qdrant bereit sind ---
        if "vision" not in models:
            raise HTTPException(status_code=503, detail="Vision-KI-Modell wurde noch nicht geladen.")
        if qdrant is None:
            raise HTTPException(status_code=503, detail="Keine Verbindung zu Qdrant möglich. Bitte Konfiguration prüfen.")

        # Datei einmalig in den Speicher lesen
        file_bytes = await file.read()
        
        # --- A. OCR Verarbeitung ---
        # Optimiertes Bild für Tesseract erstellen
        ocr_img_cv = preprocess_image_for_ocr(file_bytes)
        
        # Text extrahieren: 'deu' (Deutsch) + 'frak' (Fraktur)
        # psm 3 = Fully automatic page segmentation (Standard, meist gut)
        custom_config = r'--oem 3 --psm 3'
        extracted_text = pytesseract.image_to_string(
            ocr_img_cv, 
            lang='deu+frak', 
            config=custom_config
        )
        
        # --- B. Vektorisierung (KI) ---
        # Für CLIP nutzen wir das Originalbild (Farbe ist wichtig für Kontext)
        # Wir laden es aus den Bytes in ein PIL Image
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Vektor erstellen (Liste von gleitkommazahlen)
        # Wir nutzen das Vision-Modell.
        # Es kann sein, dass wir hier das Bild direkt übergeben können, aber als Liste ist es sicherer.
        embeddings = models["vision"].encode([pil_img])
        embedding = embeddings[0].tolist()
        
        # --- C. Speichern in Qdrant ---
        payload = {
            "filename": file.filename,
            "ocr_text": extracted_text,
            "mysql_id": mysql_id # Redundant im Payload, aber praktisch zur Anzeige
        }
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=mysql_id, # Wir nutzen die MySQL ID direkt als Vektor-ID
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        print(f"Bild {mysql_id} erfolgreich verarbeitet.")
        
        return {
            "status": "success",
            "mysql_id": mysql_id,
            "ocr_text": extracted_text,
            "message": "Bild wurde vektorisiert und Text extrahiert."
        }

    except HTTPException as he:
        # HTTPException direkt weitergeben
        raise he
    except Exception as e:
        print(f"Schwerwiegender Fehler: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
async def search_images(query: str, limit: int = 5):
    """
    Suche nach Bildern anhand von Text (z.B. "Soldat Uniform")
    """
    try:
        # --- Prüfen, ob Modell und Qdrant bereit sind ---
        if "text" not in models:
            raise HTTPException(status_code=503, detail="Text-KI-Modell wurde noch nicht geladen.")
        if qdrant is None:
            raise HTTPException(status_code=503, detail="Keine Verbindung zu Qdrant.")

        # Suchtext in Vektor wandeln mit Multilingual Modell
        search_vector = models["text"].encode(query).tolist()
        
        # Ähnliche Vektoren in Qdrant finden
        # search() ist nicht verfügbar, wir nutzen query_points()
        run_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            limit=limit
        )
        
        # Falls query_points ein Objekt mit .points zurückgibt (neue API), nutzen wir das.
        # Sonst gehen wir davon aus, dass es direkt die Liste ist.
        results = run_result.points if hasattr(run_result, 'points') else run_result
        
        return {"results": results}
    except HTTPException as he:
        raise he
    except Exception as e:
         print(f"Fehler bei Suche: {str(e)}")
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)