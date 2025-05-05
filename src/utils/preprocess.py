import os
import re
from typing import List, Dict
import markdown
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Convert multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters and normalize
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)]', ' ', text)
    return text.strip()

def extract_title_and_url(text: str) -> tuple:
    """Extract title and URL from the markdown text."""
    # Extract title
    title_match = re.search(r'## (.*?)(?:\n|$)', text)
    title = title_match.group(1) if title_match else "Untitled"
    
    # Extract URL
    url_match = re.search(r'URL: (.*?)(?:\n|$)', text)
    url = url_match.group(1) if url_match else ""
    
    return title, url

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

def process_article(article_text: str) -> List[Dict]:
    """Process a single article and return a list of document chunks."""
    # Extract title and URL
    title, url = extract_title_and_url(article_text)
    
    # Remove title and URL from content
    content = re.sub(r'## .*?\n', '', article_text)
    content = re.sub(r'URL: .*?\n', '', content)
    
    # Split content into chunks
    chunks = split_into_chunks(content)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "title": title,
            "url": url,
            "content": chunk,
            "chunk_id": f"{title}-{i}",
            "source": "angelone_support"
        }
        documents.append(doc)
    
    return documents

def process_support_data(input_file: str, output_dir: str) -> List[Dict]:
    """Process support data into document chunks suitable for embedding."""
    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split into articles
    articles = content.split("\n\n---\n\n")
    
    all_documents = []
    for article in articles:
        if article.strip():
            documents = process_article(article)
            all_documents.extend(documents)
    
    # Write processed documents to output file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_documents.txt")
    
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
    input_file = "data/angelone_support.txt"
    output_dir = "data/processed"
    
    documents = process_support_data(input_file, output_dir)
    print(f"Processed {len(documents)} document chunks from support data") 