import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

print(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    print(f"Updating collection '{COLLECTION_NAME}'...")
    from qdrant_client.models import VectorParamsDiff, Distance
    
    # We must use VectorParamsDiff for update_collection, but since we are adding a NEW vector, 
    # we might actually need to use update_collection to ADD a named vector? 
    # Wait, in qdrant 1.x, you cannot *add* a new named vector to an existing collection with update_collection reliably without the raw dict payload or update_collection(vectors_config=...) 
    # Let's try raw curl payload to add a new named vector first or just recreating if empty etc.
    # Ah, the REST api error: "Wrong input: Not existing vector name error: text"
    # Actually you cannot "add" a mixed vector config to a pure unnamed config via update_collection in Qdrant easily without recreating or passing the exact diff.
    # Let's check qdrant docs on update collection vector parameters. 
    pass
except Exception as e:
    print(f"Error updating collection: {e}")
