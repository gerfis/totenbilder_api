# ToTenBilder API

Modernes API-Backend für die semantische Suche in historischen Sterbebildern (Totenbildern).
Basierend auf **FastAPI**, **Qdrant**, **MySQL** und **FastEmbed**.

## Features

- **Semantische Suche**: Findet Bilder anhand von Textbeschreibungen ("Grabstein mit Engel") oder visueller Ähnlichkeit zu anderen Bildern.
- **High-Performance AI**: Nutzt `FastEmbed` (Quantized CLIP Models) für extrem schnelle und speicherschonende Vektorisierung.
- **Hybrid-Datenhaltung**:
  - **Vektoren**: Qdrant (für Ähnlichkeitssuche).
  - **Metadaten**: MySQL (für NID, Delta, Status).
  - **Storage**: Cloudflare R2 (S3-kompatibel).
- **Auto-Sync**: Background-Tasks zur Synchronisation von MySQL-Metadaten in den Vektor-Index.
- **Integrierte Security**: Session-basiertes Login-System und API-Key Schutz für Admin-Tasks.
- **Frontend**: Integriertes statisches Dashboard für Suche und Verwaltung.

## Voraussetzungen

### Installation

```bash
pip install -r requirements.txt
```

### Konfiguration (.env)

Erstelle eine `.env` Datei im Hauptverzeichnis:

```env
# --- Server Config ---
INDEX_API_KEY=mein_geheimer_admin_key  # Für Indexing-Endpoints

# --- Qdrant Vector DB ---
QDRANT_URL=https://...
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=totenbilder

# --- S3 / Cloudflare R2 Storage ---
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=...
R2_PREFIX=totenbilder/
R2_PUBLIC_BASE_URL=https://pub-...

# --- MySQL Datenbank ---
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=...
DB_NAME=totenbilder_db
```

## Starten

```bash
python main.py
```
Der Server läuft standardmäßig auf `http://0.0.0.0:8000`.

---

## API Referenz

### 🔍 Suche (Search)

**`GET /api/search`**
Einfache Textsuche, ideal für Browser-Tests.
- `query`: Suchbegriff
- `limit`: Anzahl (Default: 30)
- `delta`: Filter (z.B. "0", ">0", "alle")
- `type`: Such-Modus (`"image"` für visuelle Suche im Bild, `"text"` für OCR-Volltextsuche) (Default: `"image"`)

**`POST /api/search`**
Volle Suchfunktionalität inkl. Image-to-Image Suche und expliziter Textsuche.
```json
{
  "query": "Ein Soldat in Uniform",
  "type": "image",                 // Optional: "image" oder "text"
  "similar": "referenz_bild.jpg",  // Optional: Ähnlichkeitssuche anstatt Text-Query
  "limit": 50,
  "offset": 0,
  "delta": "alle"                  // Optional: "alle", "0", ">0"
}
```

### ⚙️ Indexierung (Upload/Index)

**`POST /api/index`**
Startet den Indexierungsprozess für den gesamten S3-Bucket im Hintergrund.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{ "force_reindex": false }
```

**`POST /api/index-one`**
Indiziert oder aktualisiert ein spezifisches Bild sofort.
```json
{ "filename": "1234.jpg" }
```

**`POST /api/update-text-one`**
Indiziert den Textvektor eines Bildes neu (nur bei `delta=0`). Der Bild-Hauptvektor bleibt unberührt.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{ "filename": "1234.jpg" }
```

**`POST /api/update-text-all`**
Aktualisiert alle Textvektoren für Bilder mit `delta=0` im Hintergrund. Holt die Volltexte aus der MySQL-Datenbank und speichert die neuen Vektoren in Qdrant.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{}
```

### 🗑️ Löschen (Delete)

**`POST /api/delete-by-nid`**
Löscht alle Vektoren aus Qdrant, die zu einer bestimmten personenspezifischen NID gehören.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{ "nid": 123 }
```

**`POST /api/delete-one`**
Löscht einen einzelnen Vektor anhand des Dateinamens aus dem Qdrant-Index.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{ "filename": "1234.jpg" }
```

### 🔧 Wartung (Maintenance)

**`POST /api/update-payload`**
Synchronisiert Metadaten (wie `nid`, `delta`) aus der MySQL-Datenbank in bestehende Qdrant-Vektoren.
```json
{
  "all": true   // Aktualisiert alle Einträge (Background Task)
  // ODER
  "filename": "1234.jpg" // Aktualisiert nur einen Eintrag
}
```

**`GET /api/missing-in-qdrant`**
Analyse-Tool: Vergleicht MySQL-Einträge mit dem Qdrant-Index und dem S3-Bucket.
Zeigt an:
- Bilder in DB, die in Qdrant fehlen (aber im Storage liegen -> bereit zum Indexieren).
- Bilder in DB, die nirgends existieren (Dateileichen).

### 🔐 Authentifizierung (Auth)

Die API schützt statische HTML-Seiten (`/static/*.html`) via Session-Cookie.

**`POST /auth/login`**
Login gegen Tabelle `users`.
```json
{ "username": "admin", "password": "..." }
```

**`POST /auth/logout`**
Löscht die Session.

## Projektstruktur

- `main.py`: Einstiegspunkt, Router-Konfiguration.
- `auth.py`: Login-Logik, MySQL-User-Check.
- `index.py`: Logik zum Indexieren von Bildern (S3 -> FastEmbed -> Qdrant).
- `search.py`: Suchlogik (Text/Bild -> FastEmbed -> Qdrant Search).
- `payload.py`: Synchronisation und Wartungstools.