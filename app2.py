import torch 
import streamlit as st
import os
import sys
import subprocess
import asyncio
import numpy as np
import json
import pickle
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
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Angel One Support assistant. How can I help you today?"}
    ]

st.markdown("""
<style>
    /* Global styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #333;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #ffffff;
        border-bottom: 1px solid #e9ecef;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 950px;
        margin: 0 auto;
        padding: 2.8rem;
        background-color: #ffffff;
        border-radius: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chat-container:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, #F43F5E, #E11D48, #F43F5E);
        background-size: 200% 100%;
        animation: gradientBorder 3s linear infinite;
    }
    
    @keyframes gradientBorder {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .chat-container:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.18);
        transform: translateY(-5px);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.8rem;
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        max-width: 85%;
        position: relative;
        transition: all 0.3s ease;
        z-index: 1;
    }
    
    .chat-message:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
        color: white;
        margin-left: auto;
        border-radius: 1.5rem 1.5rem 0.25rem 1.5rem;
        animation: slideInRight 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .chat-message.user:before {
        content: '';
        position: absolute;
        bottom: 0;
        right: 0;
        width: 30%;
        height: 30%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        filter: blur(15px);
        z-index: -1;
    }
    
    .chat-message.assistant {
        background-color: #F8F9FA;
        color: #1F2937;
        margin-right: auto;
        border-radius: 1.5rem 1.5rem 1.5rem 0.25rem;
        border-left: 5px solid #F43F5E;
        animation: slideInLeft 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .chat-message.assistant:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 30%;
        height: 30%;
        background: rgba(244, 63, 94, 0.08);
        border-radius: 50%;
        filter: blur(15px);
        z-index: -1;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Message content */
    .message-content {
        line-height: 1.8;
        font-size: 1.1rem;
        letter-spacing: 0.01em;
    }
    
    /* Sources section */
    .sources {
        margin-top: 1.5rem;
        padding: 1.2rem;
        border-top: 1px solid #e9ecef;
        font-size: 0.95rem;
        color: #6B7280;
        transition: all 0.4s ease;
        border-radius: 1rem;
        background-color: rgba(248, 249, 250, 0.7);
    }
    
    .sources:hover {
        background-color: #f1f5f9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    
    .sources a {
        color: #F43F5E;
        text-decoration: none;
        transition: all 0.3s;
        position: relative;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        font-weight: 500;
    }
    
    .sources a:after {
        content: '';
        position: absolute;
        width: 0;
        height: 2px;
        bottom: 0;
        left: 0;
        background-color: #E11D48;
        transition: width 0.4s ease;
    }
    
    .sources a:hover {
        color: #E11D48;
        background-color: rgba(244, 63, 94, 0.08);
    }
    
    .sources a:hover:after {
        width: 100%;
    }
    
    /* Input box */
    .stTextInput {
        position: fixed;
        bottom: 2.5rem;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 950px;
        background: white;
        padding: 1.5rem;
        border-radius: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        transition: all 0.4s ease;
        z-index: 1000;
    }
    
    .stTextInput:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.18);
        transform: translateX(-50%) translateY(-5px);
    }
    
    .stTextInput > div {
        border: 2px solid #F43F5E;
        border-radius: 1rem;
        transition: all 0.4s ease;
        overflow: hidden;
    }
    
    .stTextInput > div:before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(90deg, #F43F5E, #E11D48, #F43F5E);
        background-size: 200% 100%;
        animation: gradientBorder 3s linear infinite;
        z-index: -1;
        border-radius: 1rem;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .stTextInput > div:focus-within {
        border-color: #E11D48;
        box-shadow: 0 0 0 5px rgba(244, 63, 94, 0.25);
        transform: scale(1.02);
    }
    
    .stTextInput > div:focus-within:before {
        opacity: 1;
    }
    
    /* Sidebar styling */
    .sidebar .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
    }
    
    .sidebar .block-container:hover {
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
        transform: translateY(-3px);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 1rem;
        font-weight: 600;
        transition: all 0.4s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.3);
    }
    
    .stButton button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: all 0.8s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(244, 63, 94, 0.5);
        letter-spacing: 1.5px;
    }
    
    .stButton button:hover:before {
        left: 100%;
    }
    
    .stButton button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 10px rgba(244, 63, 94, 0.4);
    }
    
    /* Setup instructions */
    .setup-instructions {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1.5rem;
        padding: 3rem;
        margin: 3rem 0;
        border-left: 6px solid #F43F5E;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .setup-instructions:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(244, 63, 94, 0.05) 0%, transparent 70%);
        z-index: 0;
    }
    
    .setup-instructions:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        transform: translateY(-5px);
    }
    
    .setup-instructions > * {
        position: relative;
        z-index: 1;
    }
    
    /* Loading spinner */
    .stSpinner {
        border-color: #F43F5E;
    }
    
    /* Alerts and notifications */
    .stAlert {
        border-radius: 1rem;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stAlert:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    /* Code blocks */
    code {
        background-color: #f1f5f9;
        padding: 0.4em 0.6em;
        border-radius: 0.5rem;
        font-size: 0.95em;
        color: #F43F5E;
        transition: all 0.3s ease;
        border-left: 3px solid #F43F5E;
    }
    
    code:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #F43F5E 0%, #E11D48 100%);
        border-radius: 5px;
        transition: all 0.4s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #E11D48 0%, #be1a3c 100%);
    }
    
    /* App title animation */
    @keyframes gradientTitle {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .app-title {
        background: linear-gradient(90deg, #1F2937, #F43F5E, #1F2937);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: gradientTitle 6s linear infinite;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .app-title:after {
        content: '';
        position: absolute;
        width: 100px;
        height: 5px;
        background: linear-gradient(90deg, transparent, #F43F5E, transparent);
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        border-radius: 5px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #1F2937;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1;
        bottom: 150%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.4s, transform 0.4s;
        transform: translateY(10px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .tooltip .tooltiptext:after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border-width: 8px;
        border-style: solid;
        border-color: #1F2937 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
        transform: translateY(0);
    }
    
    /* Card hover effects */
    .hover-card {
        transition: all 0.4s ease;
        position: relative;
        z-index: 1;
        overflow: hidden;
    }
    
    .hover-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top right, rgba(244, 63, 94, 0.1), transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }
    
    .hover-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .hover-card:hover:before {
        opacity: 1;
    }
    
    /* Typing indicator animation */
    @keyframes typingDot {
        0% { opacity: 0.3; transform: translateY(0); }
        50% { opacity: 1; transform: translateY(-5px); }
        100% { opacity: 0.3; transform: translateY(0); }
    }
    
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        background-color: rgba(244, 63, 94, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    
    .typing-indicator span {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #F43F5E;
        margin: 0 3px;
    }
    
    .typing-indicator span:nth-child(1) {
        animation: typingDot 1s infinite 0s;
    }
    
    .typing-indicator span:nth-child(2) {
        animation: typingDot 1s infinite 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation: typingDot 1s infinite 0.4s;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        font-size: 0.95rem;
        color: #6B7280;
        border-top: 1px solid #e9ecef;
        background: linear-gradient(180deg, transparent, rgba(248, 249, 250, 0.8));
    }
    
    /* Quick suggestions */
    .quick-suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin: 2rem 0;
        justify-content: center;
    }
    
    .suggestion-chip {
        background-color: #f1f5f9;
        color: #1F2937;
        padding: 0.7rem 1.2rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .suggestion-chip:hover {
        background-color: #F43F5E;
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(244, 63, 94, 0.3);
    }
    
    /* Pulse animation for new elements */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(244, 63, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Dark mode toggle */
    .dark-mode-toggle {
        position: fixed;
        top: 1.5rem;
        right: 1.5rem;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        transition: all 0.3s ease;
    }
    
    .dark-mode-toggle:hover {
        transform: rotate(45deg);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .chat-container {
            padding: 1.5rem;
            margin: 1rem;
            border-radius: 1rem;
        }
        
        .chat-message {
            padding: 1.2rem;
            max-width: 90%;
        }
        
        .stTextInput {
            width: 95%;
            padding: 1rem;
        }
        
        .app-title {
            font-size: 2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# App title with animated gradient
# App title
st.markdown("<h1 style='text-align: center; color: #1F2937; font-size: 2.5rem; margin-bottom: 2rem;'>Angel One Support Chatbot</h1>", unsafe_allow_html=True)

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
    st.markdown("<h2 style='color: #1F2937;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
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
        st.markdown("<h4 style='color: #1F2937;'>Model Settings</h4>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='color: #1F2937;'>About</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.75rem;'>
        This chatbot uses FAISS for vector search and Gemini LLM to provide accurate answers about Angel One's services and policies.
        
        <p style='margin-top: 1rem; font-weight: 600;'>Powered by:</p>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li>🤖 Google Gemini Pro</li>
            <li>🔍 FAISS Vector Database</li>
            <li>⚡ Streamlit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# If FAISS index doesn't exist, show setup instructions
if not FAISS_INDEX_EXISTS or not DATA_EXISTS:
    st.warning("The FAISS knowledge base for this chatbot hasn't been created yet.")
    
    st.markdown("""
    <div class='setup-instructions'>
        <h3 style='color: #1F2937; margin-bottom: 1rem;'>Setup Required</h3>
        <p style='color: #4B5563; margin-bottom: 1.5rem;'>The chatbot needs to process data and create a FAISS index. This process might take a few minutes.</p>
        <ol style='color: #4B5563;'>
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
    
    st.markdown(f"<p style='text-align: center; font-size: 0.9rem; color: #6B7280; margin-bottom: 2rem;'>Using {retrieval_type} retrieval</p>", 
                unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Error initializing the retrieval system: {e}")
    if st.button("Try Creating FAISS Index Again", type="primary"):
        run_data_processing()
    st.stop()

# Main app starts here
st.markdown("<h3 style='text-align: center; color: #1F2937; margin-bottom: 2rem;'>Ask me anything about Angel One's services</h3>", unsafe_allow_html=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sources" in message:
            st.markdown(message["content"])
            if message["sources"]:
                sources_html = "<div class='sources'><p style='font-weight: 600; margin-bottom: 0.5rem;'>Sources:</p><ul style='list-style-type: none; padding-left: 0;'>"
                for source in message["sources"]:
                    sources_html += f"<li style='margin-bottom: 0.25rem;'>📄 <a href='{source['url']}' target='_blank'>{source['title']}</a></li>"
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
                sources_html = "<div class='sources'><p style='font-weight: 600; margin-bottom: 0.5rem;'>Sources:</p><ul style='list-style-type: none; padding-left: 0;'>"
                for source in result["sources"]:
                    sources_html += f"<li style='margin-bottom: 0.25rem;'>📄 <a href='{source['url']}' target='_blank'>{source['title']}</a></li>"
                sources_html += "</ul></div>"
                st.markdown(sources_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
