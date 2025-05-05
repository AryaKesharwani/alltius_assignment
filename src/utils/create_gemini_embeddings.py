import os
import re
import time
import json
import pickle
import numpy as np
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import faiss

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_gemini():
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize embedding model
    embedding_model = "models/embedding-001"
    
    return embedding_model

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text content from a TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error extracting text from {txt_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Convert multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters and normalize
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)]', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks of approximately chunk_size characters with overlap."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # If text is shorter than chunk_size, return as is
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            # Add the current chunk to chunks
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start a new chunk with overlap
            if overlap > 0 and current_chunk:
                # Get last few characters for overlap
                overlap_text = " ".join(current_chunk.split()[-overlap//10:])
                current_chunk = overlap_text + " " + sentence + " "
            else:
                current_chunk = sentence + " "
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_txt(txt_path: str) -> List[Dict]:
    """Process a single TXT file and return a list of document chunks."""
    # Extract TXT filename without extension
    txt_filename = Path(txt_path).stem
    
    # Extract text from TXT
    content = extract_text_from_txt(txt_path)
    
    if not content:
        return []
    
    # Split content into chunks
    chunks = split_into_chunks(content)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "title": txt_filename,
            "url": txt_path,
            "content": chunk,
            "chunk_id": f"{txt_filename}-{i}",
            "source": "txt"
        }
        documents.append(doc)
    
    return documents

def process_txt_file(txt_path: str) -> List[Dict]:
    if not os.path.exists(txt_path):
        print(f"TXT file not found: {txt_path}")
        return []
    
    print(f"Processing TXT file: {txt_path}")
    documents = process_txt(txt_path)
    print(f"  - Extracted {len(documents)} chunks from {Path(txt_path).name}")
    
    return documents

def generate_gemini_embeddings(documents: List[Dict], embedding_model):
    """Generate embeddings using Google's Gemini API."""
    print(f"Generating embeddings for {len(documents)} documents using Gemini...")
    embeddings = []
    
    for doc in tqdm(documents):
        try:
            # Generate embedding for the document content
            embedding = genai.embed_content(
                model=embedding_model,
                content=doc["content"],
                task_type="retrieval_document"
            )
            embeddings.append(embedding["embedding"])
        except Exception as e:
            print(f"Error generating embedding for document {doc['chunk_id']}: {e}")
            # Add a placeholder to maintain alignment with documents
            embeddings.append(None)
    
    # Filter out documents with failed embeddings
    filtered_documents = []
    filtered_embeddings = []
    
    for doc, embedding in zip(documents, embeddings):
        if embedding is not None:
            filtered_documents.append(doc)
            filtered_embeddings.append(embedding)
    
    print(f"Successfully generated {len(filtered_embeddings)} embeddings out of {len(documents)} documents")
    
    return filtered_documents, filtered_embeddings

def create_faiss_index(embeddings, index_dir: str = "faiss_index"):
    """Create a FAISS index from embeddings"""
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Get dimension of embeddings
    dimension = embeddings_array.shape[1]
    
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Create FAISS index
    # Using IndexFlatL2 for exact search with L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to the index
    index.add(embeddings_array)
    
    # Save the index
    index_path = os.path.join(index_dir, "vector.index")
    faiss.write_index(index, index_path)
    
    print(f"FAISS index created with {len(embeddings)} vectors and saved to {index_path}")
    return index

def store_metadata(documents, index_dir: str = "faiss_index"):
    """Store document metadata separately from the FAISS index"""
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Create mapping from index to document ID
    id_map = {i: doc["chunk_id"] for i, doc in enumerate(documents)}
    
    # Save id_map
    id_map_path = os.path.join(index_dir, "id_map.pkl")
    with open(id_map_path, 'wb') as f:
        pickle.dump(id_map, f)
    
    # Save documents metadata
    metadata_path = os.path.join(index_dir, "metadata.json")
    metadata = {doc["chunk_id"]: {
        "title": doc.get("title", "Untitled"),
        "url": doc.get("url", ""),
        "source": doc.get("source", "txt"),
        "content": doc.get("content", "")
    } for doc in documents}
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    return id_map, metadata

def search_index(query_text, embedding_model, top_k=5, index_dir: str = "faiss_index"):
    """Search the FAISS index with a query string"""
    # Load the index
    index_path = os.path.join(index_dir, "vector.index")
    index = faiss.read_index(index_path)
    
    # Load metadata
    id_map_path = os.path.join(index_dir, "id_map.pkl")
    with open(id_map_path, 'rb') as f:
        id_map = pickle.load(f)
    
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Generate embedding for query
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query_text,
        task_type="retrieval_query"
    )
    
    # Convert query embedding to numpy array
    query_vector = np.array([query_embedding["embedding"]], dtype=np.float32)
    
    # Search the index
    distances, indices = index.search(query_vector, top_k)
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:  # FAISS may return -1 if there are not enough results
            continue
        
        chunk_id = id_map[int(idx)]
        doc_metadata = metadata[chunk_id]
        
        results.append({
            "id": chunk_id,
            "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
            "metadata": doc_metadata
        })
    
    return results

def main():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
    try:
        embedding_model = setup_gemini()
    except Exception as e:
        print(f"Error setting up Gemini: {e}")
        return
    
    # Define TXT file path
    txt_path = "data/angelone_support.txt"
    
    # Check if TXT file exists
    if not os.path.exists(txt_path):
        print(f"TXT file not found: {txt_path}")
        return
    
    # Process TXT file
    documents = process_txt_file(txt_path)
    
    if not documents:
        print("No documents found to create embeddings. Please check if the TXT file exists and contains content.")
        return
    
    # Generate embeddings using Gemini
    filtered_documents, embeddings = generate_gemini_embeddings(documents, embedding_model)
    
    if not embeddings:
        print("Failed to generate embeddings. Please check your Gemini API key and try again.")
        return
    
    # Create FAISS index directory
    index_dir = "faiss_angelone_support"
    
    # Create FAISS index
    index = create_faiss_index(embeddings, index_dir)
    
    # Store metadata
    id_map, metadata = store_metadata(filtered_documents, index_dir)
    
    print(f"TXT embeddings successfully created with Gemini and stored in FAISS index at '{index_dir}'")
    
    # Example search (uncomment to test)
    # query = "How do I reset my password?"
    # results = search_index(query, embedding_model, top_k=3, index_dir=index_dir)
    # print(f"\nSearch results for query: '{query}'")
    # for i, result in enumerate(results):
    #     print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
    #     print(f"Title: {result['metadata']['title']}")
    #     print(f"Content: {result['metadata']['content'][:200]}...")

if __name__ == "__main__":
    main()