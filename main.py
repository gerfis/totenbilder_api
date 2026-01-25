import io
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- KONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.happyhati.com")
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

    # 2. KI-Modell laden (CLIP)
    try:
        print("HINTERGRUND: Lade KI-Modell (CLIP) 'clip-ViT-B-32-multilingual-v1'...")
        print("HINTERGRUND: Dies kann beim ersten Start (Download) mehrere Minuten dauern...")
        models["clip"] = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        print("HINTERGRUND: KI-Modell erfolgreich geladen!")
    except Exception as e:
        print(f"HINTERGRUND-FEHLER BEIM KI-MODELL: {type(e).__name__}: {str(e)}")

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

@app.get("/health")
async def health_check():
    """
    Einfacher Health-Check um zu sehen, ob der Server überhaupt läuft.
    """
    status = {
        "status": "online",
        "qdrant_connected": qdrant is not None,
        "model_loaded": "clip" in models
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
    
    # Graustufen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: Macht Hintergründe weiß und Schrift schwarz (entfernt Gilb)
    processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    return processed_img

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
        if "clip" not in models:
            raise HTTPException(status_code=503, detail="KI-Modell wurde noch nicht geladen oder konnte nicht geladen werden. Bitte Logs prüfen.")
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
        embedding = models["clip"].encode(pil_img).tolist()
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
async def search_images(query: str, limit: int = 5):
    """
    Suche nach Bildern anhand von Text (z.B. "Soldat Uniform")
    """
    try:
        # --- Prüfen, ob Modell und Qdrant bereit sind ---
        if "clip" not in models:
            raise HTTPException(status_code=503, detail="KI-Modell wurde noch nicht geladen.")
        if qdrant is None:
            raise HTTPException(status_code=503, detail="Keine Verbindung zu Qdrant.")

        # Suchtext in Vektor wandeln
        search_vector = models["clip"].encode(query).tolist()
        
        # Ähnliche Vektoren in Qdrant finden
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=search_vector,
            limit=limit
        )
        
        return {"results": results}
    except HTTPException as he:
        raise he
    except Exception as e:
         print(f"Fehler bei Suche: {str(e)}")
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)