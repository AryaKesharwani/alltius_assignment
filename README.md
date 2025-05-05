# Angel One Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot trained on Angel One customer support documentation, powered by Google's Gemini LLM and Pinecone vector database.

## Features

- 🔍 Semantic search across support documentation and PDF documents using Pinecone
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
- Pinecone API key (required)

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
   - Get a Pinecone API key from [Pinecone](https://www.pinecone.io/)
   - Add your API keys to the `.env` file:
     ```
     GOOGLE_API_KEY=your_gemini_api_key_here
     PINECONE_API_KEY=your_pinecone_api_key_here
     PINECONE_ENVIRONMENT=gcp-starter
     ```

### Data Processing

Process the data (website scraping, PDF extraction, and Pinecone vector database creation):

```
python process_data.py
```

This will:
1. Scrape Angel One support website
2. Process website data
3. Process insurance PDF files (if present)
4. Create a Pinecone vector database for semantic search

## Running the Chatbot

You can run the chatbot using one of these methods:

### Option 1: Using the startup script (Recommended)

The startup script automatically checks your environment and helps troubleshoot common issues:

```
python start.py
```

This will:
1. Check for required dependencies
2. Verify the existence of data directories
3. Check for API key configuration
4. Launch the Streamlit app with appropriate configuration

### Option 2: Direct Streamlit launch

```
streamlit run app.py
```

The chatbot will automatically open in your browser, typically at [http://localhost:8501](http://localhost:8501)

## Troubleshooting

### asyncio RuntimeError

If you encounter errors related to `asyncio` or `no running event loop`, try:

1. Use the `start.py` script which includes fixes for these issues
2. Make sure you have the latest dependencies installed:
   ```
   pip install -r requirements.txt
   ```
3. Try updating to a newer Python version (3.8 or higher recommended)

### PyTorch/TensorFlow compatibility issues

Some versions of PyTorch and TensorFlow may conflict. The project includes appropriate version constraints in the requirements file. If you encounter errors:

1. Create a fresh virtual environment
2. Install requirements with `pip install -r requirements.txt`
3. Try using `start.py` which includes compatibility checks

### Pinecone API Issues

If you encounter issues with Pinecone:

1. Ensure your API key is correctly set in the `.env` file or entered in the app's sidebar
2. Check that your Pinecone environment is correctly set (default is "gcp-starter")
3. Make sure you're using a free tier or paid plan with enough storage for your vectors

## Using Google Gemini LLM

The chatbot can use either basic retrieval or Google's Gemini LLM for enhanced responses:

### Setting up Gemini

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Enter the API key in the app's sidebar settings or add it to your `.env` file
3. Toggle "Use Gemini AI" in the Advanced Options panel

### Benefits of Gemini

- More natural, human-like responses
- Better understanding of context and complex queries
- Improved answer synthesis from multiple sources

## Deployment Options

### Option 1: Deploying on Streamlit Cloud

1. Push your code to GitHub
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository
4. Set the Gemini and Pinecone API keys as secrets in the Streamlit Cloud settings

### Option 2: Deploying on Heroku

1. Create a `Procfile` with:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. Create a `runtime.txt` file with:
   ```
   python-3.9.0
   ```

3. Deploy to Heroku:
   ```
   heroku create
   git push heroku main
   ```

4. Set environment variables:
   ```
   heroku config:set GOOGLE_API_KEY=your_gemini_api_key_here
   heroku config:set PINECONE_API_KEY=your_pinecone_api_key_here
   heroku config:set PINECONE_ENVIRONMENT=gcp-starter
   ```

## Project Structure

- `app.py` - Main Streamlit application
- `/data` - Data storage
  - `/processed` - Processed text chunks
  - `/pdfs` - PDF files
- `/src` - Source code
  - `/backend` - Backend retrieval system
    - `pinecone_retrieval.py` - Pinecone retrieval system with Gemini integration
  - `/utils` - Data processing utilities
    - `create_pinecone_embeddings.py` - Script to create Pinecone embeddings
- `extract.py` - Website scraping script
- `process_data.py` - Main data processing pipeline
- `setup.sh` - Setup script
- `example.env` - Example environment variables file

## Customization

### Modifying the Retrieval Logic

The core retrieval system is in `src/backend/pinecone_retrieval.py`. You can modify:
- The relevance threshold (default is 0.4)
- The number of results to retrieve (default is 5)
- The answer generation logic for both Gemini and basic fallback

### Adding New Data Sources

To add new data sources, create a processing script similar to `process_pdfs.py` and modify `process_data.py` to include it in the pipeline.

## Notes

- The chatbot's knowledge is limited to the content from Angel One's support website and provided PDF files.
- Pinecone requires an active internet connection to function as it is a cloud-based vector database.

## License

[MIT License](LICENSE) 