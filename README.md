# ToTenBilder API

Modernes API-Backend für die semantische Suche in historischen Sterbebildern (Totenbildern).
Basierend auf **FastAPI**, **Qdrant**, **MySQL** und der **Gemini API**.

## Features

- **Semantische Suche**: Findet Bilder anhand von Textbeschreibungen ("Grabstein mit Engel") oder visueller Ähnlichkeit zu anderen Bildern.
- **High-Performance AI**: Nutzt **Gemini 2** für modernste multimodale semantische Einbettungen (Bild+Text).
- **Datenhaltung**:
  - **Vektoren (Cloud AI)**: Qdrant Collection `totenbilder_gemini_768` (Kombiniertes Gemini 2 Embedding).
  - **Metadaten**: MySQL (für NID, Delta, Status, Roh-Feldinhalte).
  - **Storage**: Cloudflare R2 (S3-kompatibel).
- **Asymmetrische Semantische Suche**: Automatisiertes Chunking einzelner Datenbankfelder (Wohnort, Beruf, Todesgrund etc.) innerhalb des Gemini-Embeddings für exaktere Treffer.
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
QDRANT_COLLECTION_IMAGES=totenbilder_v2
QDRANT_COLLECTION_TEXTS=totenbilder_texte
QDRANT_COLLECTION_GEMINI=totenbilder_gemini_768

# --- Gemini API ---
GEMINI_API_KEY=...

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

### 🆕 Neueste Einträge (Latest)

**`GET /api/latest`**
Gibt die neuesten Totenbilder (Hauptseiten, `delta=0`) direkt aus der MySQL-Datenbank als JSON zurück, absteigend sortiert nach `nid`.
- `anzahl`: Limitiert die Anzahl der Ergebnisse (Default: 10).
- `wohnort`: Filtert die Ergebnisse nach einem bestimmten Wohnort. Ohne Angabe werden Bilder aus allen Wohnorten zurückgegeben.

**Antwort-Struktur:**
```json
[
  {
    "nid": 1234,
    "Name": "Name der Person",
    "Sterbedatum": "2023-01-01T00:00",
    "Sterbetag": 1,
    "Sterbemonat": 1,
    "Sterbejahr": 2023,
    "Wohnort": "Ortsname",
    "alias": "name-der-person",
    "url": "https://<R2_PUBLIC_BASE_URL>/totenbilder/bild.jpg"
  }
]
```

### 📅 Heutiger Todestag (Today)

**`GET /api/today`**
Gibt Totenbilder zurück, deren Sterbetag und -monat dem heutigen Datum (oder den explizit übergebenen Parametern) entsprechen. Bei Schaltjahren am 29. Februar werden spezifisch Bilder mit Todestag 29.2. zurückgeliefert. Sonst wird normal nach heutigem Tag/Monat gefiltert.
- `anzahl`: Limitiert die Anzahl der Ergebnisse (Optional, Default: Alle passenden Bilder).
- `wohnort`: Filtert die Ergebnisse nach einem bestimmten Wohnort.
- `tag`: Optionaler Tag (1-31) zur Überschreibung des heutigen Datums (muss zusammen mit `monat` verwendet werden).
- `monat`: Optionaler Monat (1-12) zur Überschreibung des heutigen Datums (muss zusammen mit `tag` verwendet werden).

*Die Antwort-Struktur ist identisch zu `/api/latest`.*

### 💻 Client-Nutzung (Frontend Integration)

Die Endpunkte `/api/latest` und `/api/today` liefern ein JSON-Array zurück, das direkt in Frontends verwendet werden kann. 

**Beispiel für die Anzeige in einer Web-App:**
- Das Bild wird direkt über die URL im Feld `url` eingebunden: `<img src="{item.url}" alt="{item.Name}" />`
- Der Link zur Detailseite der Person wird aus dem Alias generiert: `<a href="https://neue.totenbilder.at/totenbild/{item.alias}">Zum Eintrag</a>`


### 🔍 Suche (Search)

**`GET /api/search`**
Einfache Textsuche, ideal für Browser-Tests.
- `query`: Suchbegriff
- `limit`: Anzahl (Default: 30)
- `delta`: Filter (z.B. "0", ">0", "alle")
- `type`: Such-Modus (`"image"` oder `"text"`) (Default: `"image"`)
- `method`: Aktuell wird ausschließlich `"gemini"` unterstützt.

**`POST /api/search`**
Volle Suchfunktionalität inkl. Image-to-Image Suche und expliziter Textsuche via Gemini.
```json
{
  "query": "Ein Soldat in Uniform",
  "type": "image",                 // Optional: "image" oder "text"
  "method": "gemini",              // Standard: "gemini"
  "similar": "referenz_bild.jpg",  // Optional: Ähnlichkeitssuche anstatt Text-Query
  "limit": 50,
  "offset": 0,
  "delta": "alle"                  // Optional: "alle", "0", ">0"
}
```

**Antwort-Struktur (Rückgabe):**
Beispiel-Antwort für beide `/api/search` Methoden. Bei `type="text"` liefert die API zusätzlich die Felder `field_type` (z.B. "Todesgrund") und `text_content`, die dem Frontend exakt zeigen, *warum* und *wo* der Suchbegriff gefunden wurde.

```json
[
  {
    "filename": "musterbild.jpg",
    "image_url": "https://pub-...",
    "score": 0.85,
    "nid": 1234,
    "delta": 0,
    "field_type": "Todesgrund",           // NEU (nur bei type="text")
    "text_content": "gefallen in Russland" // NEU (nur bei type="text")
  }
]
```

### ⚙️ Indexierung (Upload/Index)

**`POST /api/index`**
Startet den Indexierungsprozess für den gesamten S3-Bucket im Hintergrund.
*Header:* `X-API-Key: <INDEX_API_KEY>`
```json
{ "force_reindex": false }
```

**`POST /api/index-one`**
Indiziert oder aktualisiert ein spezifisches Bild sofort (Gemini 2).
```json
{ "filename": "1234.jpg" }
```

**`POST /api/index-all-gemini`**
Startet die exklusive Neu-Indexierung aller Bilder (Bild+Text Kombination) mittels der **Gemini Embedding 2** API im Hintergrund. Die Collection wird vorher zurückgesetzt.
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
- `index.py`: Logik zum Indexieren von Bildern (S3 -> Gemini API -> Qdrant).
- `search.py`: Suchlogik (Text/Bild -> Gemini API -> Qdrant Search).
- `payload.py`: Synchronisation und Wartungstools.