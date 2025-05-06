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
    """Configure and return the Gemini embedding model."""
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize embedding model
    embedding_model = "models/embedding-001"
    
    return embedding_model

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        from pypdf import PdfReader
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
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

def process_pdf(pdf_path: str) -> List[Dict]:
    """Process a single PDF file and return a list of document chunks."""
    # Extract PDF filename without extension
    filename = Path(pdf_path).stem
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(f"No text extracted from {pdf_path}")
        return []
    
    # Split text into chunks
    chunks = split_into_chunks(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "title": filename,
            "content": chunk,
            "chunk_id": f"{filename}-{i}",
            "source": pdf_path
        }
        documents.append(doc)
    
    return documents

def process_pdf_directory(pdf_dir: str) -> List[Dict]:
    """Process all PDF files in a directory and return a list of document chunks."""
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        documents = process_pdf(pdf_file)
        all_documents.extend(documents)
    
    return all_documents

def generate_gemini_embeddings(documents: List[Dict], embedding_model: str) -> tuple:
    """Generate embeddings for documents using Gemini API."""
    embeddings = []
    filtered_documents = []
    
    print(f"Generating embeddings for {len(documents)} document chunks...")
    
    for doc in tqdm(documents):
        try:
            # Generate embedding
            result = genai.embed_content(
                model=embedding_model,
                content=doc["content"],
                task_type="retrieval_document"
            )
            
            # Add embedding to list
            embeddings.append(result["embedding"])
            filtered_documents.append(doc)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error generating embedding for document {doc['chunk_id']}: {e}")
    
    return filtered_documents, embeddings

def create_faiss_index(embeddings: List[List[float]], index_dir: str) -> faiss.Index:
    """Create a FAISS index from embeddings."""
    # Convert embeddings to numpy array
    embedding_array = np.array(embeddings, dtype=np.float32)
    
    # Get dimensions
    num_embeddings, dim = embedding_array.shape
    
    # Create index
    index = faiss.IndexFlatL2(dim)
    
    # Add vectors to index
    index.add(embedding_array)
    
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Save index
    index_path = os.path.join(index_dir, "vector.index")
    faiss.write_index(index, index_path)
    
    print(f"Created FAISS index with {num_embeddings} vectors of dimension {dim}")
    print(f"Index saved to {index_path}")
    
    return index

def store_metadata(documents: List[Dict], index_dir: str) -> tuple:
    """Store document metadata and ID mapping."""
    # Create ID map (position in index -> document ID)
    id_map = {i: doc["chunk_id"] for i, doc in enumerate(documents)}
    
    # Create metadata map (document ID -> metadata)
    metadata = {doc["chunk_id"]: {
        "title": doc["title"],
        "content": doc["content"],
        "source": doc["source"]
    } for doc in documents}
    
    # Save ID map
    id_map_path = os.path.join(index_dir, "id_map.pkl")
    with open(id_map_path, 'wb') as f:
        pickle.dump(id_map, f)
    
    # Save metadata
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"ID map saved to {id_map_path}")
    print(f"Metadata saved to {metadata_path}")
    return id_map, metadata

def search_index(query_text, embedding_model, top_k=5, index_dir: str = "faiss_pdf_index"):
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
    
    pdf_dir = "data/pdfs"
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return
    
    # Process PDF files
    documents = process_pdf_directory(pdf_dir)
    
    if not documents:
        print("No documents found to create embeddings. Please check if the PDF directory contains PDF files.")
        return
    
    # Generate embeddings using Gemini
    filtered_documents, embeddings = generate_gemini_embeddings(documents, embedding_model)
    
    if not embeddings:
        print("Failed to generate embeddings. Please check your Gemini API key and try again.")
        return
    
    index_dir = "faiss_pdf_index"
    
    # Create FAISS index
    index = create_faiss_index(embeddings, index_dir)
    
    # Store metadata
    id_map, metadata = store_metadata(filtered_documents, index_dir)
    
    print(f"PDF embeddings successfully created with Gemini and stored in FAISS index at '{index_dir}'")
    
    # Example search (uncomment to test)
    # query = "What is the main topic of these documents?"
    # results = search_index(query, embedding_model, top_k=3, index_dir=index_dir)
    # print(f"\nSearch results for query: '{query}'")
    # for i, result in enumerate(results):
    #     print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
    #     print(f"Title: {result['metadata']['title']}")
    #     print(f"Content: {result['metadata']['content'][:200]}...")

if __name__ == "__main__":
    main()
