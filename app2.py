import torch 
import streamlit as st
import os
import sys
import subprocess
import asyncio
import numpy as np
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import faiss

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Load environment variables
load_dotenv()


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Fix for asyncio event loop error
def fix_event_loop():
    """Ensure a running event loop is available"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Create a new event loop if one is not running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Call the fix at startup
fix_event_loop()

# FAISS Retrieval System Class
class FAISSRetrievalSystem:
    def __init__(self, index_dir="faiss_angelone_support", model_name="gemini-1.5-pro"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.is_initialized = False
        self.initialize()
        
    def initialize(self):
        """Initialize the FAISS retrieval system"""
        try:
            # Check if the index exists
            index_path = os.path.join(self.index_dir, "vector.index")
            id_map_path = os.path.join(self.index_dir, "id_map.pkl")
            metadata_path = os.path.join(self.index_dir, "metadata.json")
            
            if not os.path.exists(index_path) or not os.path.exists(id_map_path) or not os.path.exists(metadata_path):
                print(f"FAISS index files not found in {self.index_dir}")
                self.is_initialized = False
                return
            
            # Load the index
            self.index = faiss.read_index(index_path)
            
            # Load id_map
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Initialize Gemini API if a key is provided
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                genai.configure(api_key=google_api_key)
                # Get available models to confirm API is working
                try:
                    self.embedding_model = "models/embedding-001"
                    genai.get_model(self.model_name)
                    self.gemini_available = True
                    print(f"Successfully initialized Gemini API with model {self.model_name}")
                except Exception as e:
                    print(f"Error initializing Gemini API: {e}")
                    self.gemini_available = False
            else:
                self.gemini_available = False
                
            self.is_initialized = True
            print("FAISS retrieval system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing FAISS retrieval system: {e}")
            self.is_initialized = False
            
    def search(self, query, top_k=5):
        """Search the FAISS index with a query string"""
        if not self.is_initialized:
            return [], []
            
        try:
            # Generate embedding for query
            query_embedding = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            
            # Convert query embedding to numpy array
            query_vector = np.array([query_embedding["embedding"]], dtype=np.float32)
            
            # Search the index
            distances, indices = self.index.search(query_vector, top_k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # FAISS may return -1 if there are not enough results
                    continue
                
                chunk_id = self.id_map[int(idx)]
                doc_metadata = self.metadata[chunk_id]
                
                results.append({
                    "id": chunk_id,
                    "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
                    "metadata": doc_metadata
                })
            
            return results
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return []
    
    def answer_query(self, query, num_results=5):
        """Generate an answer for a query using retrieved context"""
        if not self.is_initialized:
            return {
                "answer": "I'm sorry, the retrieval system is not initialized properly. Please check your setup.",
                "has_answer": False,
                "sources": []
            }
            
        search_results = self.search(query, top_k=num_results)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "has_answer": False,
                "sources": []
            }
            
        # Prepare context from search results
        context = "\n\n".join([result["metadata"]["content"] for result in search_results])
        sources = [{
            "title": result["metadata"]["title"],
            "url": result["metadata"]["url"],
            "content": result["metadata"]["content"][:200] + "..." if len(result["metadata"]["content"]) > 200 else result["metadata"]["content"]
        } for result in search_results]
        
        if self.gemini_available:
            try:
                # Use Gemini to generate a response
                gemini_model = genai.GenerativeModel(self.model_name)
                
                prompt = f"""
                You are a helpful assistant for Angel One support. Use only the following context to answer the user's question.
                If the answer is not in the context, say "I don't have information about that in my knowledge base."
                
                Context:
                {context}
                
                User question: {query}
                
                Answer in a helpful, concise way:
                """
                
                response = gemini_model.generate_content(prompt)
                answer = response.text
                
                return {
                    "answer": answer,
                    "has_answer": True,
                    "sources": sources
                }
            except Exception as e:
                print(f"Error generating response with Gemini: {e}")
                
                # Fallback to basic response if Gemini fails
                answer = f"I found some information that might help:\n\n"
                for result in search_results[:3]:
                    answer += f"• {result['metadata']['content'][:300]}...\n\n"
                
                return {
                    "answer": answer,
                    "has_answer": True,
                    "sources": sources
                }
        else:
            # Basic response without Gemini
            answer = f"I found some information that might help:\n\n"
            for result in search_results[:3]:
                answer += f"• {result['metadata']['content'][:300]}...\n\n"
            
            return {
                "answer": answer,
                "has_answer": True,
                "sources": sources
            }

# Function to run data processing script
def run_data_processing():
    """Run the data processing pipeline"""
    with st.spinner("Running data processing..."):
        try:
            # Run the custom FAISS processing script
            process = subprocess.Popen(
                [sys.executable, "create_faiss_index.py"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                st.success("Data processing completed successfully! Please refresh the page.")
                return True
            else:
                st.error(f"Error during data processing: {stderr}")
                return False
        except Exception as e:
            st.error(f"Error running data processing: {e}")
            return False


# Page configuration
st.set_page_config(
    page_title="Angel One Support Chatbot",
    page_icon="🤖",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Angel One Support assistant. How can I help you today?"}
    ]

# Custom CSS for styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #F43F5E;
        color: white;
        border-radius: 0.5rem 0.5rem 0 0.5rem;
        align-self: flex-end;
    }
    .chat-message.assistant {
        background-color: #F5F7FB;
        color: #333;
        border-radius: 0.5rem 0.5rem 0.5rem 0;
        align-self: flex-start;
    }
    .chat-message .message-content {
        display: flex;
        flex-direction: column;
    }
    .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        background-color: #ccc;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.2rem;
        margin-right: 1rem;
    }
    .sources {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        font-style: italic;
    }
    .setup-instructions {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 4px solid #f43f5e;
    }
    .stTextInput {
        position: fixed;
        bottom: 3rem;
        background-color: white;
        padding: 1rem 0;
        width: 100%;
        left: 0;
    }
    .main-container {
        margin-bottom: 6rem;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    /* Improved button styling */
    .stButton button {
        width: 100%;
        background-color: #F43F5E;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #E11D48;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Model selector styling */
    .model-selector {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Angel One Support Chatbot")

# Define the FAISS index paths
FAISS_INDEX_DIR = "faiss_angelone_support"
DATA_PATH = "data/angelone_support.txt"
FAISS_INDEX_EXISTS = (
    os.path.exists(os.path.join(FAISS_INDEX_DIR, "vector.index")) and
    os.path.exists(os.path.join(FAISS_INDEX_DIR, "id_map.pkl")) and
    os.path.exists(os.path.join(FAISS_INDEX_DIR, "metadata.json"))
)
DATA_EXISTS = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0

# Sidebar for API key configuration and advanced options
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key configuration
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    with st.expander("API Key Settings", expanded=not bool(api_key)):
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            value=api_key,
            type="password",
            help="Enter your Gemini API key to enable AI powered responses."
        )
        
        if gemini_api_key and gemini_api_key != api_key:
            # Save API key to .env file
            with open(".env", "w") as f:
                f.write(f"GOOGLE_API_KEY={gemini_api_key}\n")
            st.success("API key saved! Please refresh the app.")
            
    # Advanced options
    with st.expander("Advanced Options"):
        st.markdown("#### Model Settings")
        use_gemini = st.toggle("Use Gemini AI for Responses", value=bool(api_key), disabled=not bool(api_key))
        st.session_state["use_gemini"] = use_gemini
        
        if not api_key and use_gemini:
            st.warning("Please enter a valid Gemini API key to enable AI features.")
        
        model_version = st.selectbox(
            "Model Version",
            options=["gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            disabled=not bool(api_key)
        )
        st.session_state["model_version"] = model_version
        
        num_results = st.slider("Number of documents to retrieve", min_value=3, max_value=10, value=5)
        st.session_state["num_results"] = num_results
        
    # Reset chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Angel One Support assistant. How can I help you today?"}
        ]
        st.rerun()
        
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses FAISS for vector search and Gemini LLM to provide accurate answers about Angel One's services and policies.
    
    **Powered by:**
    - Google Gemini Pro
    - FAISS Vector Database
    - Streamlit
    """)

# If FAISS index doesn't exist, show setup instructions
if not FAISS_INDEX_EXISTS or not DATA_EXISTS:
    st.warning("The FAISS knowledge base for this chatbot hasn't been created yet.")
    
    st.markdown("""
    <div class='setup-instructions'>
        <h3>Setup Required</h3>
        <p>The chatbot needs to process data and create a FAISS index. This process might take a few minutes.</p>
        <ol>
            <li>Make sure your text data is in the <code>data/angelone_support.txt</code> file</li>
            <li>Click the button below to create the FAISS index</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Create FAISS Index", type="primary"):
        run_data_processing()
    
    # Display manual instructions
    with st.expander("Manual Setup Instructions"):
        st.code("""
        # Run these commands in your terminal:
        
        # 1. Make sure you have installed all requirements
        pip install -r requirements.txt
        
        # 2. Run the FAISS index creation script
        python create_faiss_index.py
        
        # 3. Refresh this page when done
        """)
    
    # Create a sample create_faiss_index.py file
    if st.button("Generate FAISS Index Script"):
        script_content = """import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import json
import pickle
import re
from tqdm import tqdm

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_gemini():
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize embedding model
    embedding_model = "models/embedding-001"
    
    return embedding_model

def extract_text_from_txt(txt_path: str) -> str:
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error extracting text from {txt_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    # Convert multiple spaces to single space
    text = re.sub(r'\\s+', ' ', text)
    # Remove special characters and normalize
    text = re.sub(r'[^\\w\\s\\.\,\\?\\!\\:\\;\\-\\(\\)]', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # If text is shorter than chunk_size, return as is
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\\s+', cleaned_text)
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

def process_txt(txt_path: str):
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

def process_txt_file(txt_path: str):
    if not os.path.exists(txt_path):
        print(f"TXT file not found: {txt_path}")
        return []
    
    print(f"Processing TXT file: {txt_path}")
    documents = process_txt(txt_path)
    print(f"  - Extracted {len(documents)} chunks from {Path(txt_path).name}")
    
    return documents

def generate_gemini_embeddings(documents, embedding_model):
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

if __name__ == "__main__":
    main()
"""
        with open("create_faiss_index.py", "w") as f:
            f.write(script_content)
        st.success("Successfully created 'create_faiss_index.py' file. You can now run the index creation.")
    
    st.stop()  # Stop execution here until setup is complete

# If we got here, FAISS index exists and we can initialize the retrieval system
try:
    # Ensure event loop is properly set
    fix_event_loop()
    
    # Determine which model to use based on API key
    if os.getenv("GOOGLE_API_KEY") and st.session_state.get("use_gemini", True):
        model_version = st.session_state.get("model_version", "gemini-pro")
    else:
        model_version = None
    
    @st.cache_resource
    def get_retrieval_system(model_name=None):
        try:
            return FAISSRetrievalSystem(
                index_dir=FAISS_INDEX_DIR,
                model_name=model_name
            )
        except Exception as e:
            st.error(f"Error initializing FAISS retrieval system: {e}")
            return None
    
    retrieval_system = get_retrieval_system(model_version)
    
    if retrieval_system and retrieval_system.is_initialized:
        if retrieval_system.gemini_available:
            retrieval_type = "FAISS + Gemini AI"
        else:
            retrieval_type = "FAISS (Basic Answers)"
    else:
        st.error("Failed to initialize the FAISS retrieval system.")
        st.stop()
    
    st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>Using {retrieval_type} retrieval</p>", 
                unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Error initializing the retrieval system: {e}")
    if st.button("Try Creating FAISS Index Again", type="primary"):
        run_data_processing()
    st.stop()

# Main app starts here
st.markdown("<h3 style='text-align: center;'>Ask me anything about Angel One's services</h3>", unsafe_allow_html=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sources" in message:
            st.markdown(message["content"])
            if message["sources"]:
                sources_html = "<div class='sources'><p>Sources:</p><ul>"
                for source in message["sources"]:
                    sources_html += f"<li><a href='{source['url']}' target='_blank'>{source['title']}</a></li>"
                sources_html += "</ul></div>"
                st.markdown(sources_html, unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# User input 
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Get response from retrieval system
        with st.spinner("Thinking..."):
            num_results = st.session_state.get("num_results", 5)
            result = retrieval_system.answer_query(prompt, num_results=num_results)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "sources": result["sources"] if result["has_answer"] else []
            })
            
            # Display the response
            message_placeholder.markdown(result["answer"])
            
            # Display sources if available
            if result["has_answer"] and result["sources"]:
                sources_html = "<div class='sources'><p>Sources:</p><ul>"
                for source in result["sources"]:
                    sources_html += f"<li><a href='{source['url']}' target='_blank'>{source['title']}</a></li>"
                sources_html += "</ul></div>"
                st.markdown(sources_html, unsafe_allow_html=True)

