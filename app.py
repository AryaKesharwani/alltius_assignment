import streamlit as st
import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Page configuration
st.set_page_config(
    page_title="Angel One Support Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Initialize session state to store chat history
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
</style>
""", unsafe_allow_html=True)

# App title
st.title("Angel One Support Chatbot")

# Check if the data processing has been done
DB_PATH = "data/vectordb"
DATA_PATH = "data/angelone_support.txt"
VECTORDB_EXISTS = os.path.exists(DB_PATH) and os.path.isdir(DB_PATH) and os.listdir(DB_PATH)
DATA_EXISTS = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0

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

# If vector database doesn't exist, show processing instructions
if not VECTORDB_EXISTS or not DATA_EXISTS:
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

# Now try to import the retrieval system (only after we know the database exists)
try:
    from src.backend.retrieval import RetrievalSystem
    
    # Initialize the retrieval system
    @st.cache_resource
    def get_retrieval_system():
        return RetrievalSystem(DB_PATH)

    retrieval_system = get_retrieval_system()
    
    if not retrieval_system.is_initialized:
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
st.markdown("Ask me anything about Angel One's services", unsafe_allow_html=True)

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

# Add a footer with instructions
st.markdown("""
---
### How to use this chatbot
- Ask any question about Angel One's services or policies
- The chatbot will search through Angel One support documentation to find answers
- If information isn't found in the knowledge base, the chatbot will respond with "I don't know"

### If you're not getting good responses
If the chatbot isn't giving good answers, you might need to rebuild the knowledge base:
1. Add any additional PDF files to the `data/pdfs` directory
2. Run `python process_data.py` to rebuild the knowledge base
3. Refresh this page
""") 