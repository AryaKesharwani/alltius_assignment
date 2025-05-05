import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_gemini():
    """Configure Google Generative AI with API key."""
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
        sys.exit(1)
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize embedding model
    embedding_model = "models/embedding-001"
    
    return embedding_model

def test_gemini_embeddings():
    print("Testing Google Gemini embeddings...")
    
    # Setup Gemini
    embedding_model = setup_gemini()
    
    # Sample text to embed
    sample_texts = [
        "How do I open a trading account with Angel One?",
        "What are the charges for equity trading?",
        "How to withdraw funds from my Angel One account?",
        "What documents are needed for KYC verification?",
        "How to place a stop loss order in Angel One?"
    ]
    
    # Generate embeddings
    print("\nGenerating embeddings for sample texts...")
    for text in tqdm(sample_texts):
        try:
            # Generate embedding for document
            doc_embedding = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            # Generate embedding for query
            query_embedding = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type="retrieval_query"
            )
            
            # Print dimension info
            print(f"\nText: '{text}'")
            print(f"Document embedding dimension: {len(doc_embedding['embedding'])}")
            print(f"Query embedding dimension: {len(query_embedding['embedding'])}")
            
            # Print a few values to verify
            print(f"First 5 values (document): {doc_embedding['embedding'][:5]}")
            print(f"First 5 values (query): {query_embedding['embedding'][:5]}")
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
    
    print("\nEmbedding test completed.")

if __name__ == "__main__":
    test_gemini_embeddings() 