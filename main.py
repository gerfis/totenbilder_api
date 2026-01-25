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
# In Coolify heißt der Service oft einfach "qdrant" oder du nutzt die interne IP
# Fallback auf localhost für lokales Testen
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "totenbilder"

# Globale Variablen (werden beim Start befüllt)
models = {}
qdrant = None

import asyncio

# --- LIFESPAN (Startup & Shutdown Logik) ---

async def initialize_app():
    """
    Diese Funktion läuft im Hintergrund, damit der Server sofort startet.
    """
    global qdrant
    
    # 1. Verbindung zu Qdrant
    try:
        print(f"HINTERGRUND: Versuche Verbindung zu Qdrant unter: {QDRANT_URL}")
        # Wir setzen einen expliziten Timeout von 60 Sekunden für die Client-Anfragen
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Testen der Verbindung
        collections = qdrant.get_collections()
        print(f"HINTERGRUND: Erfolgreich mit Qdrant verbunden. Vorhandene Collections: {[c.name for c in collections.collections]}")

        # Collection erstellen, falls sie noch nicht existiert
        if not qdrant.collection_exists(COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"HINTERGRUND: Collection '{COLLECTION_NAME}' neu erstellt.")
        else:
            print(f"HINTERGRUND: Collection '{COLLECTION_NAME}' gefunden.")
    except Exception as e:
        print(f"HINTERGRUND-FEHLER BEI QDRANT: {str(e)}")
        qdrant = None

    # 2. KI-Modell laden (CLIP Multilingual)
    try:
        print("HINTERGRUND: Lade KI-Modell (CLIP)... Dies kann einige Minuten dauern...")
        # Lade in einem Thread, um den Eventpool nicht zu blockieren
        loop = asyncio.get_event_loop()
        models["clip"] = await loop.run_in_executor(None, lambda: SentenceTransformer('clip-ViT-B-32-multilingual-v1'))
        print("HINTERGRUND: KI-Modell erfolgreich geladen!")
    except Exception as e:
        print(f"HINTERGRUND-FEHLER BEIM KI-MODELL: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Wird beim Start des Servers ausgeführt.
    Startet die Initialisierung im Hintergrund.
    """
    # Startet die Initialisierung, ohne den Boot-Vorgang zu blockieren
    asyncio.create_task(initialize_app())
    
    print("API-Server wird gestartet... (KI-Initialisierung läuft im Hintergrund)")
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

    except Exception as e:
        print(f"Fehler: {str(e)}")
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
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)