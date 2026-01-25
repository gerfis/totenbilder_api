# Basis-Image (z.B. Python)
FROM python:3.11-slim

# 1. System-Pakete installieren
# tesseract-ocr: Die Engine
# tesseract-ocr-deu: Modernes Deutsch
# tesseract-ocr-frak: Fraktur (Altdeutsch)
# libgl1: Wird oft für OpenCV benötigt, falls du Bildvorverarbeitung machst
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-frak \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Arbeitsverzeichnis erstellen
WORKDIR /app

# 3. Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Restlichen Code kopieren
COPY . .

# Start-Befehl (z.B. deine API oder Worker)
CMD ["python", "main.py"]