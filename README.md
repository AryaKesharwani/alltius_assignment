# Angel One Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot trained on Angel One customer support documentation, powered by Google's Gemini LLM and FAISS vector database.

## Features

- 🔍 Semantic search across support documentation and PDF documents using FAISS
- 🧠 Google Gemini LLM for improved context understanding and natural responses
- 🤖 Answers user queries based on Angel One support information
- 🔗 Provides source links to original documentation
- 🚫 Responds with "I don't know" for questions outside of its knowledge base
- 💻 Clean, responsive user interface built with Streamlit

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Pip (Python package manager)
- Internet connection to download dependencies and scrape website data
- Google Gemini API key (optional but recommended)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

   Alternatively, you can perform these steps manually:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add any insurance PDF files to the `data/pdfs` directory

4. Set up API keys:
   - Copy `example.env` to `.env`
   - Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Add your API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```

### Data Processing

Process the data (website scraping, PDF extraction, and FAISS vector database creation):