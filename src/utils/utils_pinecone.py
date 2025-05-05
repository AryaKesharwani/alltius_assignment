import os
from dotenv import load_dotenv
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def create_pinecone_index(index_name: str, dimension: int = 768):
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable not set.")
        return False
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        if index_name in [idx.name for idx in existing_indexes]:
            print(f"Index '{index_name}' already exists. Reusing it.")
            return True
        
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be initialized
        while index_name not in [idx.name for idx in pc.list_indexes()]:
            print(f"Waiting for index '{index_name}' to be initialized...")
            time.sleep(1)
            
        print(f"Created new index '{index_name}'")
        return True
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return False