import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
OLD_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")
NEW_COLLECTION = f"{OLD_COLLECTION}_v2"

print(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    print(f"Creating new collection '{NEW_COLLECTION}'...")
    client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config={
            "": VectorParams(size=512, distance=Distance.COSINE),
            "text": VectorParams(size=384, distance=Distance.COSINE)
        }
    )
    
    # Create payload indices
    client.create_payload_index(collection_name=NEW_COLLECTION, field_name="filename", field_schema="keyword")
    client.create_payload_index(collection_name=NEW_COLLECTION, field_name="nid", field_schema="integer")
    client.create_payload_index(collection_name=NEW_COLLECTION, field_name="delta", field_schema="integer")
    
    print("New collection created successfully.")
    print("Please update QDRANT_COLLECTION_NAME in .env to point to the new collection, then run index.py or index_text.py to populate it.")
    
except Exception as e:
    print(f"Error: {e}")
