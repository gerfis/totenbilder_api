import os
import sys
import argparse
import mysql.connector
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors
from fastembed import TextEmbedding
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")

R2_PREFIX = os.getenv("R2_PREFIX", "totenbilder/")

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
        print(f"Error connecting to database: {err}", file=sys.stderr)
        sys.exit(1)

def get_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

def get_text_model():
    print(f"Loading MiniLM TEXT Model (fastembed)...")
    return TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_metadata(filename=None):
    """
    Returns list of dicts: [{'filename': ..., 'nid': ..., 'delta': ..., 'Fulltext': ...}]
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # We only care about delta=0 and non-empty Fulltext 
    query = """
        SELECT b.filename, b.nid, b.delta, t.Fulltext 
        FROM totenbilder_bilder b 
        LEFT JOIN totenbilder t ON b.nid = t.nid 
        WHERE b.delta = 0 AND t.Fulltext IS NOT NULL AND t.Fulltext != ''
    """
    params = ()
    
    if filename:
        db_filename = filename.replace(R2_PREFIX, "")
        query += " AND (b.filename = %s OR b.filename = %s)"
        params = (db_filename, filename)
        
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return results

def process_single(filename):
    results = get_metadata(filename)
    if not results:
        print(f"No suitable metadata found for '{filename}' (must be delta=0 with valid Fulltext).", file=sys.stderr)
        return

    row = results[0]
    qdrant = get_qdrant_client()
    model_txt = get_text_model()
    
    full_key = f"{R2_PREFIX}{row['filename']}" if not row['filename'].startswith(R2_PREFIX) else row['filename']
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, full_key))
    
    try:
        text_vector = list(model_txt.embed([row['Fulltext']]))[0].tolist()
        qdrant.update_vectors(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointVectors(
                    id=point_id,
                    vector={"text": text_vector}
                )
            ]
        )
        print(f"Successfully updated text vector for {full_key}")
    except Exception as e:
        print(f"Error updating text vector for {full_key}: {e}", file=sys.stderr)

def process_all():
    print("Fetching records from database...", file=sys.stderr)
    results = get_metadata()
    print(f"Found {len(results)} valid records suitable for text indexing.", file=sys.stderr)
    
    if not results:
        return
        
    qdrant = get_qdrant_client()
    model_txt = get_text_model()
    
    success_count = 0
    error_count = 0
    
    batch_points = []
    
    # Pre-compute UUIDs to ensure match with index.py
    for row in tqdm(results, desc="Updating Qdrant Text Vectors"):
        raw_filename = row['filename']
        full_key = f"{R2_PREFIX}{raw_filename}" if not raw_filename.startswith(R2_PREFIX) else raw_filename
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, full_key))
        
        try:
            text_vector = list(model_txt.embed([row['Fulltext']]))[0].tolist()
            batch_points.append(
                PointVectors(
                    id=point_id,
                    vector={"text": text_vector}
                )
            )
            success_count += 1
            
            # Batch upload every 100 points
            if len(batch_points) >= 100:
                qdrant.update_vectors(
                    collection_name=QDRANT_COLLECTION,
                    points=batch_points
                )
                batch_points = []
                
        except Exception as e:
            error_count += 1
            # tqdm.write(f"Error: {e}")

    # Upload remaining
    if batch_points:
        try:
            qdrant.update_vectors(
                collection_name=QDRANT_COLLECTION,
                points=batch_points
            )
        except Exception as e:
            print(f"Error on final batch upload: {e}")
            error_count += len(batch_points)
            success_count -= len(batch_points)

    print(f"\nCompleted.", file=sys.stderr)
    print(f"Success: {success_count}", file=sys.stderr)
    print(f"Errors: {error_count}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Update text vectors in Qdrant from MySQL Fulltext.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filename", help="The filename of the image (e.g., 'image.jpg')")
    group.add_argument("--alle", action="store_true", help="Update ALL delta=0 records from the database")
    
    args = parser.parse_args()
    
    if args.filename:
        process_single(args.filename)
    elif args.alle:
        process_all()

if __name__ == "__main__":
    main()
