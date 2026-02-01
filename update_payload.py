import os
import sys
import argparse
import mysql.connector
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Configuration
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
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


def update_qdrant_point(qdrant_client, filename, nid, delta):
    """
    Updates a single point in Qdrant. returns True on success, False on failure.
    """
    full_key = f"{R2_PREFIX}{filename}"
    
    try:
        search_result = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=full_key))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = search_result
        if not points:
            # Silent failure for bulk vs verbose for single? 
            # We'll return False so caller can decide/count.
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


def process_single(filename):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT nid, delta FROM totenbilder_bilder WHERE filename = %s"
    cursor.execute(query, (filename,))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if not result:
        print(f"Error: Filename '{filename}' not found in database.", file=sys.stderr)
        sys.exit(1)

    qdrant = get_qdrant_client()
    success, msg = update_qdrant_point(qdrant, filename, result['nid'], result['delta'])
    
    if success:
        full_key = f"{R2_PREFIX}{filename}"
        print(f"Successfully updated payload for {full_key}: nid={result['nid']}, delta={result['delta']}")
    else:
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)


def process_all():
    print("Fetching all records from database...", file=sys.stderr)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT filename, nid, delta FROM totenbilder_bilder"
    cursor.execute(query)
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    print(f"Found {len(results)} records. connecting to Qdrant...", file=sys.stderr)
    qdrant = get_qdrant_client()
    
    success_count = 0
    error_count = 0
    skipped_count = 0 # Not found in Qdrant
    
    # Using tqdm for progress bar
    for row in tqdm(results, desc="Updating Qdrant"):
        filename = row['filename']
        nid = row['nid']
        delta = row['delta']
        
        success, msg = update_qdrant_point(qdrant, filename, nid, delta)
        
        if success:
            success_count += 1
        else:
            if "Vector not found" in msg:
                skipped_count += 1
            else:
                error_count += 1
                # Optionally print actual errors?
                # tqdm.write(f"Error updating {filename}: {msg}")

    print(f"\nCompleted.", file=sys.stderr)
    print(f"Success: {success_count}", file=sys.stderr)
    print(f"Skipped (not in Qdrant): {skipped_count}", file=sys.stderr)
    print(f"Errors: {error_count}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Update metadata payload in Qdrant from MySQL.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--filename", help="The filename of the image (e.g., 'image.jpg')")
    group.add_argument("--alle", action="store_true", help="Update ALL records from the database")
    
    args = parser.parse_args()
    
    if args.filename:
        process_single(args.filename)
    elif args.alle:
        process_all()


if __name__ == "__main__":
    main()
