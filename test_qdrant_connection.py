
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"Connecting to Qdrant at: {QDRANT_URL}")
# print(f"API Key: {QDRANT_API_KEY}")

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections()
    print("Successfully connected!")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Failed to connect: {e}")
