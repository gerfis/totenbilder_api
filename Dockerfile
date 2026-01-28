# Basis-Image (z.B. Python)
FROM python:3.11-slim-bookworm

# 1. System-Pakete installieren
# Minimale Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]