import os
import re
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_processed_documents(input_file: str) -> List[Dict]:
    """Load processed documents from file."""
    if not os.path.exists(input_file):
        print(f"Warning: Input file {input_file} does not exist.")
        return []
        
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        document_blocks = content.split("\n---\n\n")
        documents = []
        
        for block in document_blocks:
            if not block.strip():
                continue
                
            doc = {}
            title_match = re.search(r'TITLE: (.*?)(?:\n|$)', block)
            url_match = re.search(r'URL: (.*?)(?:\n|$)', block)
            content_match = re.search(r'CONTENT: (.*?)(?:\n|$)', block, re.DOTALL)
            chunk_id_match = re.search(r'CHUNK_ID: (.*?)(?:\n|$)', block)
            source_match = re.search(r'SOURCE: (.*?)(?:\n|$)', block)
            
            if title_match:
                doc["title"] = title_match.group(1)
            if url_match:
                doc["url"] = url_match.group(1)
            if content_match:
                doc["content"] = content_match.group(1)
            if chunk_id_match:
                doc["chunk_id"] = chunk_id_match.group(1)
            if source_match:
                doc["source"] = source_match.group(1)
                
            if "content" in doc and doc["content"].strip():
                if "chunk_id" not in doc or not doc["chunk_id"]:
                    # Generate a chunk ID if one doesn't exist
                    title = doc.get("title", "unknown")
                    doc["chunk_id"] = f"{title}-{len(documents)}"
                documents.append(doc)
        
        return documents
    except Exception as e:
        print(f"Error loading documents from {input_file}: {e}")
        return []

def create_pinecone_index(index_name: str, dimension: int = 384):
    """Create a new Pinecone index if it doesn't exist."""
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable not set.")
        return False
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index already exists
        if index_name in pc.list_indexes():
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
        while not index_name in pc.list_indexes():
            print(f"Waiting for index '{index_name}' to be initialized...")
            time.sleep(1)
            
        print(f"Created new index '{index_name}'")
        return True
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return False

def create_pinecone_embeddings(documents: List[Dict], index_name: str = "angelone-support"):
    """Create Pinecone embeddings from documents."""
    if not documents:
        print("No documents to create embeddings for. Aborting.")
        return False
        
    try:
        # Initialize the model for embeddings
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        
        # Get dimension of the embeddings
        sample_embedding = model.encode("Sample text")
        dimension = len(sample_embedding)
        
        # Create or get Pinecone index
        if not create_pinecone_index(index_name, dimension):
            return False
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Check if index is empty and delete all vectors if not
        stats = index.describe_index_stats()
        if stats.get('total_vector_count', 0) > 0:
            print(f"Found {stats['total_vector_count']} existing vectors in index '{index_name}'. Deleting them...")
            index.delete(delete_all=True)
            print("All vectors deleted from the index")
        
        # Prepare data for batch insertion
        batch_size = 50  # Smaller batch size to avoid rate limits
        
        print(f"Creating embeddings for {len(documents)} documents in {(len(documents)-1)//batch_size + 1} batches")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Ensure unique IDs
            ids = []
            for j, doc in enumerate(batch):
                chunk_id = doc["chunk_id"]
                # Make sure the ID is unique by appending an index if needed
                while chunk_id in ids:
                    chunk_id = f"{doc['chunk_id']}-{j}"
                ids.append(chunk_id)
            
            contents = [doc["content"] for doc in batch]
            
            # Create embeddings for the batch
            print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            embeddings = model.encode(contents, show_progress_bar=True)
            
            # Prepare records for Pinecone
            records = []
            for j, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
                doc = batch[j]
                metadata = {
                    "title": doc.get("title", "Untitled"),
                    "url": doc.get("url", ""),
                    "source": doc.get("source", "unknown"),
                    "content": doc.get("content", "")
                }
                record = {
                    "id": doc_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                }
                records.append(record)
            
            # Add to Pinecone
            print(f"Adding batch {i//batch_size + 1} to Pinecone")
            index.upsert(vectors=records)
            
            # Sleep a bit to avoid rate limits
            if i + batch_size < len(documents):
                time.sleep(1)
        
        # Verify insertion
        stats = index.describe_index_stats()
        print(f"Added {stats['total_vector_count']} vectors to Pinecone index '{index_name}'")
        
        return True
    except Exception as e:
        print(f"Error creating Pinecone embeddings: {e}")
        return False

def main():
    # Define file paths
    processed_documents_file = "data/processed/processed_documents.txt"
    processed_pdfs_file = "data/processed/processed_pdfs.txt"
    combined_file = "data/processed/combined_documents.txt"
    
    # Check if combined file exists, if not create it
    if not os.path.exists(combined_file):
        print("Combined document file not found, creating it from individual sources")
        
        combined_documents = []
        
        # Load website documents
        website_documents = load_processed_documents(processed_documents_file)
        if website_documents:
            print(f"Loaded {len(website_documents)} documents from website")
            combined_documents.extend(website_documents)
        
        # Load PDF documents
        pdf_documents = load_processed_documents(processed_pdfs_file)
        if pdf_documents:
            print(f"Loaded {len(pdf_documents)} documents from PDFs")
            combined_documents.extend(pdf_documents)
            
        # Write combined documents to file
        if combined_documents:
            with open(combined_file, "w", encoding="utf-8") as f:
                for doc in combined_documents:
                    f.write(f"TITLE: {doc['title']}\n")
                    f.write(f"URL: {doc['url']}\n")
                    f.write(f"CONTENT: {doc['content']}\n")
                    f.write(f"CHUNK_ID: {doc['chunk_id']}\n")
                    f.write(f"SOURCE: {doc['source']}\n")
                    f.write("\n---\n\n")
            print(f"Created combined document file with {len(combined_documents)} documents")
    
    # Load documents from combined file
    documents = load_processed_documents(combined_file)
    
    if documents:
        print(f"Loaded {len(documents)} documents from combined file")
        
        # Check if Pinecone API key is set
        if not PINECONE_API_KEY:
            print("Error: PINECONE_API_KEY environment variable not set. Please set it before running this script.")
            print("You can get a free API key from https://www.pinecone.io/")
            return
        
        # Create Pinecone embeddings
        if create_pinecone_embeddings(documents):
            print("Pinecone vector database created successfully")
        else:
            print("Failed to create Pinecone vector database")
    else:
        print("No documents found to create embeddings. Please check if data extraction was successful.")
        print("You can run the data extraction again with: python extract.py")

if __name__ == "__main__":
    main() 