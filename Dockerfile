# Basis-Image (z.B. Python)
FROM python:3.11-slim-bookworm

# 1. System-Pakete installieren
# tesseract-ocr: Die Engine
# tesseract-ocr-deu: Modernes Deutsch
# tesseract-ocr-frak: Fraktur (Altdeutsch)
# libgl1: Wird oft für OpenCV benötigt, falls du Bildvorverarbeitung machst
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-deu \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    libgomp1 \
    && mkdir -p /usr/share/tesseract-ocr/tessdata \
    && curl -L -o /usr/share/tesseract-ocr/tessdata/frak.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/frak.traineddata \
    && rm -rf /var/lib/apt/lists/*

# 2. Arbeitsverzeichnis erstellen
WORKDIR /app

# 3. Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Restlichen Code kopieren
COPY . .

# Start-Befehl (z.B. deine API oder Worker)
# Port freigeben
EXPOSE 8000

# Start-Befehl: Uvicorn Server starten
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]