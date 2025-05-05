import os
import re
from pathlib import Path
from typing import List, Dict
import pypdf

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from each page
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
    """Process a single PDF and return a list of document chunks."""
    # Extract PDF filename without extension
    pdf_filename = Path(pdf_path).stem
    
    # Extract text from PDF
    content = extract_text_from_pdf(pdf_path)
    
    if not content:
        return []
    
    # Split content into chunks
    chunks = split_into_chunks(content)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "title": pdf_filename,
            "url": pdf_path,
            "content": chunk,
            "chunk_id": f"{pdf_filename}-{i}",
            "source": "insurance_pdf"
        }
        documents.append(doc)
    
    return documents

def process_pdfs(pdf_dir: str, output_file: str) -> List[Dict]:
    """Process all PDFs in a directory and save to output file."""
    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                 if f.lower().endswith('.pdf')]
    
    all_documents = []
    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        documents = process_pdf(pdf_path)
        all_documents.extend(documents)
        print(f"  - Extracted {len(documents)} chunks")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write processed documents to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(f"TITLE: {doc['title']}\n")
            f.write(f"URL: {doc['url']}\n")
            f.write(f"CONTENT: {doc['content']}\n")
            f.write(f"CHUNK_ID: {doc['chunk_id']}\n")
            f.write(f"SOURCE: {doc['source']}\n")
            f.write("\n---\n\n")
    
    return all_documents

if __name__ == "__main__":
    pdf_dir = "data/pdfs"
    output_file = "data/processed/processed_pdfs.txt"
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created directory {pdf_dir} - please place PDF files here")
    else:
        documents = process_pdfs(pdf_dir, output_file)
        print(f"Processed {len(documents)} document chunks from PDFs") 