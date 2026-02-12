import os
import mysql.connector
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
R2_PREFIX = os.getenv("R2_PREFIX", "totenbilder/")

router = APIRouter()

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def get_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

def update_qdrant_point(qdrant_client, filename, nid, delta):
    """
    Updates a single point in Qdrant. returns True on success, False on failure.
    """
    # Assuming filename in DB is just "foo.jpg", but in Qdrant it might be "prefix/foo.jpg"
    # Logic from update_payload.py:
    # full_key = f"{R2_PREFIX}{filename}"
    
    # However, sometimes R2_PREFIX might already be included or not needed depending on how it was indexed.
    # The original script assumes it needs to prepend R2_PREFIX.
    
    full_key = f"{R2_PREFIX}{filename}"
    
    try:
        # Search for the point by filename in payload
        # Note: We are scrolling to find the point ID by the payload field 'filename'
        points, _ = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=full_key))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            return False, f"Vector not found for key '{full_key}'"
            
        point_id = points[0].id
        
        # Update payload
        qdrant_client.set_payload(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point_id],
            payload={
                "nid": nid,
                "delta": delta
            }
        )
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def process_payload_update(filename: str = None, all_records: bool = False):
    """
    Background task logic
    """
    print(f"Starting Payload Update. Single: {filename}, All: {all_records}")
    
    conn = get_db_connection()
    if not conn:
        print("DB Connection failed")
        return

    qdrant = get_qdrant_client()
    if not qdrant:
        print("Qdrant Connection failed")
        conn.close()
        return

    cursor = conn.cursor(dictionary=True)
    
    success_count = 0
    error_count = 0
    skipped_count = 0

    try:
        if filename:
            query = "SELECT filename, nid, delta FROM totenbilder_bilder WHERE filename = %s"
            cursor.execute(query, (filename,))
            rows = cursor.fetchall() # Should be 1
        else:
            query = "SELECT filename, nid, delta FROM totenbilder_bilder"
            cursor.execute(query)
            rows = cursor.fetchall()
            
        total = len(rows)
        print(f"Found {total} rows to process.")
        
        for row in rows:
            fname = row['filename']
            nid = row['nid']
            delta = row['delta']
            
            success, msg = update_qdrant_point(qdrant, fname, nid, delta)
            
            if success:
                success_count += 1
            else:
                if "Vector not found" in msg:
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"Error updating {fname}: {msg}")
                    
        print(f"Payload Update Completed. Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count}")

    except Exception as e:
        print(f"Error in process_payload_update: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# API Models
class PayloadUpdateRequest(BaseModel):
    filename: str | None = None
    all: bool = False

@router.post("/update-payload")
async def trigger_payload_update(request: PayloadUpdateRequest, background_tasks: BackgroundTasks):
    """
    Triggers the payload update process (syncing metadata from MySQL to Qdrant).
    """
    if not request.filename and not request.all:
        raise HTTPException(status_code=400, detail="Either filename or all=True must be provided")

    background_tasks.add_task(process_payload_update, request.filename, request.all)
    
    mode = "ALL records" if request.all else f"Single file: {request.filename}"
    return {"message": f"Payload update started in background for {mode}"}

import boto3

# Add Env Variables for R2 (Redundant to index.py but keeps payload isolated)
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
# R2_PREFIX is already defined above

def get_s3_client():
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("WARNUNG: R2 Credentials fehlen für Payload-Check!")
        return None
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )

@router.get("/missing-in-qdrant")
def check_missing_in_qdrant():
    """
    Vergleicht die 'totenbilder_bilder' Tabelle mit dem Qdrant Index UND prüft den R2 Bucket.
    Gibt zurück:
    - missing_qdrant_ok_r2: In MySQL, fehlt in Qdrant, aber in R2 vorhanden (Ready to Index)
    - missing_qdrant_missing_r2: In MySQL, fehlt in Qdrant, und AUCH in R2 nicht gefunden (Dateileiche in DB)
    """
    conn = get_db_connection()
    qdrant = get_qdrant_client()
    s3 = get_s3_client()
    
    if not conn or not qdrant:
        raise HTTPException(status_code=500, detail="Datenbank oder Qdrant nicht verfügbar.")
    
    try:
        # 1. Alle Dateinamen aus MySQL holen
        cursor = conn.cursor(dictionary=True)
        print("Lade Dateinamen aus MySQL...")
        cursor.execute("SELECT filename FROM totenbilder_bilder")
        mysql_rows = cursor.fetchall()
        
        mysql_filenames = set()
        for row in mysql_rows:
            fname = row['filename']
            # Normalisierung: Wir erwarten im Bucket "totenbilder/dateiname.jpg"
            # In DB steht oft nur "dateiname.jpg". 
            # Wir bauen hier den Key so zusammen, wie wir ihn in Qdrant UND S3 erwarten.
            
            # Fix: Falls fname schon mit dem Prefix beginnt, nicht nochmal davor hängen!
            if fname.startswith(R2_PREFIX):
                full_key = fname
            else:
                full_key = f"{R2_PREFIX}{fname}"
            
            mysql_filenames.add(full_key)
            
        print(f"MySQL enthält {len(mysql_filenames)} Bilder (Keys).")

        # 2. Alle Dateinamen aus Qdrant holen
        print("Lade Dateinamen aus Qdrant (Scroll)...")
        qdrant_filenames = set()
        
        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                with_payload=["filename"],
                with_vectors=False,
                limit=1000,
                offset=offset
            )
            for point in points:
                if point.payload and "filename" in point.payload:
                    qdrant_filenames.add(point.payload["filename"])
            
            offset = next_offset
            if offset is None:
                break
        
        print(f"Qdrant enthält {len(qdrant_filenames)} Bilder.")
        
        # 3. Abgleich: Was fehlt in Qdrant?
        missing_in_qdrant_set = mysql_filenames - qdrant_filenames
        missing_list = list(missing_in_qdrant_set)
        
        print(f"Fehlen in Qdrant: {len(missing_list)}. Prüfe nun R2 Bucket...")

        # 4. R2 Prüfung
        # Wir laden ALLE Objekte im Prefix, um nicht N Requests zu machen.
        # Das ist bei < 50k Objekten meist schneller als N Head-Requests.
        r2_files_set = set()
        
        if s3:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix=R2_PREFIX)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        r2_files_set.add(obj['Key'])
        else:
             print("WARNUNG: S3 Client nicht verfügbar, R2 Check übersprungen.")

        # 5. Kategorisierung
        ok_r2 = []       # In MySQL, fehlt in Qdrant, aber in R2 da -> Indexierung starten!
        missing_r2 = []  # In MySQL, fehlt in Qdrant, und fehlt R2 -> DB Bereinigung nötig?
        
        for key in missing_list:
            if key in r2_files_set:
                ok_r2.append(key)
            else:
                missing_r2.append(key)

        return {
            "total_mysql": len(mysql_filenames),
            "total_qdrant": len(qdrant_filenames),
            "total_missing_in_qdrant": len(missing_list),
            
            "ready_to_index_count": len(ok_r2),
            "missing_in_r2_count": len(missing_r2),
            
            "ready_to_index_files": ok_r2[:500],
            "missing_in_r2_files": missing_r2[:500]
        }

    except Exception as e:
        print(f"Fehler beim Abgleich: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
