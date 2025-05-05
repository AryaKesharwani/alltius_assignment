import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import requests
        import bs4
        import pinecone
        import sentence_transformers
        import pypdf
        import streamlit
        import google.generativeai
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    return True

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data",
        "data/processed",
        "data/pdfs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def main():
    """Main function to run the entire data processing pipeline."""
    if not check_dependencies():
        return
    
    create_directories()
    
    # Step 1: Extract data from Angel One support website
    if not os.path.exists("data/angelone_support.txt") or os.path.getsize("data/angelone_support.txt") == 0:
        if not run_script("extract.py", "Extracting data from Angel One support website"):
            print("Failed to extract website data. Please check the extract.py script.")
            return
    else:
        print("\nUsing existing extracted data from Angel One support website")
    
    # Step 2: Check for PDF files
    pdf_dir = "data/pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"\nNo PDF files found in {pdf_dir}. Please add your insurance PDFs to this directory.")
        print("You can continue with just the website data, but for best results, add PDF files.")
    
    # Step 3: Process website data
    if not run_script("src/utils/preprocess.py", "Processing website data"):
        print("Failed to process website data. Please check the preprocess.py script.")
        return
    
    # Step 4: Process PDF files if they exist
    if pdf_files:
        if not run_script("src/utils/process_pdfs.py", "Processing PDF files"):
            print("Failed to process PDF files. Please check the process_pdfs.py script.")
            return
    
    # Step 5: Combine processed data
    processed_files = [
        "data/processed/processed_documents.txt",
        "data/processed/processed_pdfs.txt"
    ]
    
    combined_file = "data/processed/combined_documents.txt"
    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file_path in processed_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
    
    print(f"\nCombined processed data into {combined_file}")
    
    # Step 6: Ask the user whether to use Gemini or sentence-transformers for embeddings
    use_gemini = input("\nDo you want to use Google Gemini for PDF embeddings? (y/n): ").lower().strip() == 'y'
    
    if use_gemini:
        # Check for Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("\nWarning: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
            print("You can get a free API key from https://aistudio.google.com/app/apikey")
            return
        
        # Create PDF embeddings with Gemini
        if not run_script("src/utils/create_gemini_embeddings.py", "Creating PDF embeddings with Gemini and storing in Pinecone"):
            print("Failed to create Gemini embeddings. Please check the create_gemini_embeddings.py script.")
            return
    else:
        # Create Pinecone vector database using sentence-transformers
        if not run_script("src/utils/create_pinecone_embeddings.py", "Creating Pinecone vector database with sentence-transformers"):
            print("Failed to create Pinecone vector database. Please check the create_pinecone_embeddings.py script.")
            return
    
    print("\n\nData processing complete! You can now run the chatbot with: streamlit run app.py")

if __name__ == "__main__":
    main() 