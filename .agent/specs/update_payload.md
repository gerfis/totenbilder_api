# Task: Erstelle ein Python Skript, das die Payload in Qdrant aktualisiert.

## Context
Wir erweitern unser Backend um ein Modul (`update_payload.py`), das aus einer MySQL Datenbank (Tabelle `totenbilder_bilder`) die Metadaten (`nid`, `delta`) zu einem Bild ausliest und dann in Qdrant den entsprechenden Vektor-Punkt aktualisiert.

## Requirements

### 1. Umgebung & Konfiguration
* **Dateipfad:** `update_payload.py`
* **Library:** 
    * `mysql-connector-python` (muss zu `requirements.txt` hinzugefügt werden).
    * `qdrant-client` (bereits vorhanden).
    * `python-dotenv` (bereits vorhanden).
* **Environment Variablen (.env):**
    * MySQL: `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
    * Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`
    * S3 Prefix: `R2_PREFIX` (z.B. `totenbilder/`)

### 2. Logik
Der Prozess läuft wie folgt ab:

1.  **Parameter:** Das Skript akzeptiert:
    *   `--filename` (z.B. `schernhammer_1.jpg`) für einzelne Updates.
    *   `--alle` (Flag) um ALLE Einträge aus der Datenbank zu verarbeiten.
2.  **MySQL Lookup:**
    *   Verbindung zur MySQL Datenbank herstellen.
    *   Einzelmodus: `SELECT nid, delta FROM totenbilder_bilder WHERE filename = %s`
    *   Bulk-Modus (`--alle`): `SELECT filename, nid, delta FROM totenbilder_bilder`
    *   Wenn kein Eintrag gefunden wird: Fehler ausgeben und beenden.
3.  **Qdrant Lookup:**
    *   Verbindung zu Qdrant herstellen.
    *   Der Key für Qdrant setzt sich zusammen aus `R2_PREFIX` + `filename` (z.B. `totenbilder/schernhammer_1.jpg`).
    *   Suche in der Collection (`QDRANT_COLLECTION_NAME` a.k.a. `totenbilder`) nach dem Punkt, dessen Payload-Feld `filename` exakt diesem Key entspricht.
        *   Benutze `scroll()` mit Filter: `FieldCondition(key="filename", match=MatchValue(value=full_key))`.
    *   Wenn kein Punkt gefunden wird: Fehler ausgeben ("Vector not found for key ...").
4.  **Qdrant Update:**
    *   Extrahiere die Point-ID (UUID) aus dem Suchergebnis.
    *   Aktualisiere die Payload dieses Punktes mit `nid` und `delta`.
        *   `set_payload(collection_name=..., points=[id], payload={"nid": nid, "delta": delta})`
5.  **Output:**
    *   Erfolg: "Successfully updated payload for {full_key}: nid={nid}, delta={delta}"
    *   Fehler: Entsprechende Fehlermeldung auf stderr.

### 3. Error Handling
*   `ValueError` wenn `--filename` fehlt.
*   Logge Fehler bei Datenbankverbindung oder Qdrant-Timeouts.

## Definition of Done
* [ ] `requirements.txt` enthält `mysql-connector-python`.
* [ ] `update_payload.py` ist implementiert und ausführbar.
* [ ] Unit-Tests in `tests/test_update_payload.py` (optional, falls Test-Infrastruktur steht, sonst Manual Test).
* [ ] Der Code entspricht dem PEP8 Standard.
