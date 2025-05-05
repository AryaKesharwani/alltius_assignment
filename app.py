import torch 
import streamlit as st
import os
import sys
import subprocess
import asyncio
from pathlib import Path
from dotenv import load_dotenv

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Load environment variables
load_dotenv()

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

# Check if the data processing has been done
DATA_PATH = "data/angelone_support.txt"
PROCESSED_DATA_PATH = "data/processed/combined_documents.txt"
DATA_EXISTS = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0
PROCESSED_DATA_EXISTS = os.path.exists(PROCESSED_DATA_PATH) and os.path.getsize(PROCESSED_DATA_PATH) > 0

# Sidebar for API key configuration and advanced options
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key configuration
    api_key = os.getenv("GOOGLE_API_KEY", "")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    
    with st.expander("API Key Settings", expanded=not bool(api_key) or not bool(pinecone_api_key)):
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            value=api_key,
            type="password",
            help="Enter your Gemini API key to enable AI powered responses."
        )
        
        pinecone_api_key_input = st.text_input(
            "Pinecone API Key", 
            value=pinecone_api_key,
            type="password",
            help="Enter your Pinecone API key for vector database."
        )
        
        pinecone_env = st.text_input(
            "Pinecone Environment", 
            value=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
            help="Enter your Pinecone environment (e.g., gcp-starter)"
        )
        
        if gemini_api_key and gemini_api_key != api_key or pinecone_api_key_input and pinecone_api_key_input != pinecone_api_key or pinecone_env and pinecone_env != os.getenv("PINECONE_ENVIRONMENT"):
            # Save API keys (this is for demonstration; in production, handle secrets more securely)
            env_content = []
            if gemini_api_key:
                env_content.append(f"GOOGLE_API_KEY={gemini_api_key}")
            if pinecone_api_key_input:
                env_content.append(f"PINECONE_API_KEY={pinecone_api_key_input}")
            if pinecone_env:
                env_content.append(f"PINECONE_ENVIRONMENT={pinecone_env}")
                
            with open(".env", "w") as f:
                f.write("\n".join(env_content) + "\n")
            st.success("API keys saved! Please refresh the app.")
            
    # Advanced options
    with st.expander("Advanced Options"):
        st.markdown("#### Model Settings")
        use_gemini = st.toggle("Use Gemini AI for Responses", value=bool(api_key), disabled=not bool(api_key))
        st.session_state["use_gemini"] = use_gemini
        
        # Add new toggle for Gemini embeddings
        use_gemini_embeddings = st.toggle(
            "Use Gemini for PDF Embeddings", 
            value=False, 
            disabled=not bool(api_key),
            help="Use Google Gemini to generate embeddings for PDFs instead of sentence-transformers"
        )
        st.session_state["use_gemini_embeddings"] = use_gemini_embeddings
        
        if not api_key and use_gemini:
            st.warning("Please enter a valid Gemini API key to enable AI features.")
        
        model_version = st.selectbox(
            "Model Version",
            options=["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
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
    This chatbot uses Pinecone and Gemini LLM to provide accurate answers about Angel One's services and policies.
    
    **Powered by:**
    - Google Gemini Pro
    - Pinecone Vector Database
    - Streamlit
    """)

def run_data_processing():
    """Run the data processing pipeline"""
    with st.spinner("Running data processing..."):
        try:
            # Run the data processing script
            process = subprocess.Popen(
                [sys.executable, "process_data.py"], 
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

# If data hasn't been processed, show processing instructions
if not DATA_EXISTS or not PROCESSED_DATA_EXISTS:
    st.warning("The knowledge base for this chatbot hasn't been created yet.")
    
    st.markdown("""
    <div class='setup-instructions'>
        <h3>Setup Required</h3>
        <p>The chatbot needs to process data from Angel One's support website and PDFs. This process might take a few minutes.</p>
        <ol>
            <li>Make sure any insurance PDFs are in the <code>data/pdfs</code> directory</li>
            <li>Click the button below to start data processing</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Run Data Processing", type="primary"):
        run_data_processing()
    
    # Display manual instructions as well
    with st.expander("Manual Setup Instructions"):
        st.code("""
        # Run these commands in your terminal:
        
        # 1. Make sure you have installed all requirements
        pip install -r requirements.txt
        
        # 2. Run the data processing script
        python process_data.py
        
        # 3. Refresh this page when done
        """)
    
    st.stop()  # Stop execution here until setup is complete

# Now try to import the retrieval system (only after we know the data exists)
try:
    # Ensure event loop is properly set before importing models
    fix_event_loop()
    
    # Check if Pinecone API key is set
    if not os.getenv("PINECONE_API_KEY"):
        st.error("Pinecone API Key is not set. Please set it in the sidebar.")
        st.stop()
    
    # Determine which retrieval system to use based on API key
    if os.getenv("GOOGLE_API_KEY") and st.session_state.get("use_gemini", True):
        try:
            from src.backend.pinecone_retrieval import PineconeRetrievalSystem as RetrievalSystemClass
            model_version = st.session_state.get("model_version", "gemini-pro")
            
            @st.cache_resource
            def get_retrieval_system():
                try:
                    # Always use angelone-support as the default index name
                    index_name = "angelone-support"
                    # Get embeddings choice from session state
                    use_gemini_emb = st.session_state.get("use_gemini_embeddings", False)
                    
                    # Return properly configured retrieval system
                    return RetrievalSystemClass(
                        index_name=index_name,
                        model_name=model_version,
                        use_gemini_embeddings=use_gemini_emb
                    )
                except Exception as e:
                    st.error(f"Error initializing Pinecone+Gemini retrieval system: {e}")
                    return None
                
            retrieval_system = get_retrieval_system()
            # Check if we got the correct retrieval system type
            if retrieval_system and retrieval_system.is_initialized:
                retrieval_type = "Pinecone + Gemini AI"
            else:
                st.error("Failed to initialize the Pinecone retrieval system with Gemini.")
                st.stop()
                
        except ImportError as e:
            st.error(f"Could not import Pinecone+Gemini modules: {e}")
            st.stop()
    else:
        # If Gemini API key not set, can still use Pinecone with basic answer generation
        try:
            from src.backend.pinecone_retrieval import PineconeRetrievalSystem as RetrievalSystemClass
            
            @st.cache_resource
            def get_retrieval_system():
                try:
                    # Always use angelone-support as the index name
                    index_name = "angelone-support"
                    # Get embeddings choice from session state
                    use_gemini_emb = st.session_state.get("use_gemini_embeddings", False)
                    
                    system = RetrievalSystemClass(
                        index_name=index_name,
                        use_gemini_embeddings=use_gemini_emb
                    )
                    # Even if Gemini isn't available, Pinecone retrieval should work
                    if system.is_initialized:
                        return system
                    return None
                except Exception as e:
                    st.error(f"Error initializing Pinecone retrieval system: {e}")
                    return None
                
            retrieval_system = get_retrieval_system()
            
            if retrieval_system and retrieval_system.is_initialized:
                retrieval_type = "Pinecone (Basic Answers)"
            else:
                st.error("Failed to initialize the Pinecone retrieval system.")
                st.stop()
                
        except ImportError as e:
            st.error(f"Could not import Pinecone modules: {e}")
            st.stop()
    
    st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>Using {retrieval_type} retrieval</p>", unsafe_allow_html=True)
    
    if not retrieval_system or not retrieval_system.is_initialized:
        st.error("Could not initialize the retrieval system. The data may not have been processed correctly.")
        if st.button("Try Running Data Processing Again", type="primary"):
            run_data_processing()
        st.stop()
        
except Exception as e:
    st.error(f"Error initializing the retrieval system: {e}")
    if st.button("Try Running Data Processing Again", type="primary"):
        run_data_processing()
    st.stop()

# If we got here, everything is initialized correctly
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
            result = retrieval_system.answer_query(prompt)
            
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

# # Add a footer with instructions
# st.markdown("""
# ---
# ### How to use this chatbot
# - Ask any question about Angel One's services or policies
# - The chatbot will search through Angel One support documentation to find answers
# - If information isn't found in the knowledge base, the chatbot will respond with "I don't know"

# ### Enhance with Gemini AI
# For better context understanding and more natural responses:
# 1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
# 2. Enter the API key in the sidebar settings
# 3. Refresh the page to activate Gemini-powered responses

# ### If you're not getting good responses
# If the chatbot isn't giving good answers, you might need to rebuild the knowledge base:
# 1. Add any additional PDF files to the `data/pdfs` directory
# 2. Run `python process_data.py` to rebuild the knowledge base
# 3. Refresh this page
# """) 