import faiss
import numpy as np
import pickle
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ====== Load environment and configure Gemini ======
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=GEMINI_API_KEY)

# ====== Configuration ======
FAISS_INDEX_PATH = "faiss_pdf_index/vector.index"
ID_MAP_PATH = "faiss_pdf_index/id_map.pkl"
METADATA_PATH = "faiss_pdf_index/metadata.json"
TOP_K = 5

# ====== Load FAISS index and metadata ======
index = faiss.read_index(FAISS_INDEX_PATH)

with open(ID_MAP_PATH, "rb") as f:
    id_map = pickle.load(f)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# ====== Embed query using Gemini Embedding API ======
def embed_with_gemini(query: str) -> np.ndarray:
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return np.array(response["embedding"], dtype=np.float32).reshape(1, -1)

# ====== Search FAISS index ======
def search_faiss(query: str, top_k: int = TOP_K):
    query_vector = embed_with_gemini(query)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        doc_id = id_map.get(idx)
        doc_meta = metadata.get(doc_id, {})
        results.append({
            "id": doc_id,
            "title": doc_meta.get("title", "Unknown"),
            "content": doc_meta.get("content", "[No content found]"),
            "source": doc_meta.get("source", "N/A"),
            "score": distances[0][rank]
        })
    return results

# ====== Ask Gemini with context from top documents ======
def ask_gemini_with_context(query: str, retrieved_docs: list[dict]) -> str:
    context = "\n\n".join(
        f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)
    )

    prompt = (
        "You are a helpful assistant. Answer the user's question based only on the context from the documents below.\n\n"
        f"Context:\n{context}\n\n"
        f"User's question: {query}\n"
    )

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text

# ====== Exported helper function ======
def run_semantic_search(query: str, top_k: int = TOP_K):
    results = search_faiss(query, top_k=top_k)
    answer = ask_gemini_with_context(query, results)
    return {
        "query": query,
        "results": results,
        "answer": answer
    }
