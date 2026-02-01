# ToTenBilder API

API-Backend für die semantische Suche in Bildern, basierend auf FastAPI, Qdrant und CLIP.

## Features

- **Semantische Suche**: Findet Bilder anhand von Textbeschreibungen oder ähnlichen Bildern.
- **Auto-Indexing**: Indiziert Bilder aus einem S3-kompatiblen Bucket (z.B. Cloudflare R2).
- **Hybrid-Architektur**: Nutzt `clip-ViT-B-32` für Bild-Embeddings und `clip-ViT-B-32-multilingual-v1` für Text-Queries.

## Voraussetzungen

Stelle sicher, dass eine `.env` Datei mit folgenden Variablen existiert:

```env
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=...
R2_ENDPOINT_URL=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=...
R2_PREFIX=...
R2_PUBLIC_BASE_URL=...
INDEX_API_KEY=mein_geheimer_api_key
```

## API Endpoints

### 🔍 Suche

**`POST /api/search`**

Sucht nach Bildern basierend auf Text oder einem Referenzbild.

**Body (`json`):**
```json
{
  "query": "Ein Grabstein aus Granit",
  "similar": null,   // Optional: Dateiname eines Bildes für Ähnlichkeitssuche
  "limit": 30,       // Optional: Standard 30
  "offset": 0        // Optional: Standard 0 (für Pagination)
}
```

**Antwort:**
Eine Liste von Ergebnissen mit `filename`, `image_url` und `score`.

**`GET /api/search`**

Ermöglicht die Text-Suche per GET-Request (z.B. für Browser-Tests).

**Parameter:**
- `query`: Suchtext (z.B. `?query=Baum`).
- `limit`: (Optional) Anzahl der Ergebnisse (Standard: 30).
- `offset`: (Optional) Offset für Pagination (Standard: 0).

**Beispiel:**
`GET /api/search?query=Grabstein`

**Antwort:**
Identisch zu `POST /api/search`.

---

### ⚙️ Indexierung

**`POST /api/index`**

Startet den Indexierungsprozess für den gesamten konfigurierten Bucket im Hintergrund.
Prüft existierende Bilder und überspringt diese (außer `force_reindex` ist aktiv).

**Benötigt Header:**
`X-API-Key: <INDEX_API_KEY>`

**Body (`json`):**
```json
{
  "force_reindex": false
}
```

**`POST /api/index-one`**

Indiziert ein spezifisches Bild sofort.

**Body (`json`):**
```json
{
  "filename": "ordner/bild.jpg"
}
```

---

### ❤️ Health Check

**`GET /health`**

Prüft, ob der Service läuft.
```json
{ "status": "ok" }
```