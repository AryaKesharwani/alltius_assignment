import asyncio
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Fix for asyncio event loop issues
def ensure_event_loop():
    """Ensure a running event loop is available"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Create a new event loop if one is not running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

class PineconeRetrievalSystem:
    def __init__(self, index_name: str = "angelone-support", 
                 model_name: str = "gemini-pro", 
                 use_gemini_embeddings: bool = False,
                 embedding_model_name: str = "paraphrase-MiniLM-L6-v2"):
        """Initialize the Pinecone-powered retrieval system with Gemini.
        
        Args:
            index_name: Name of the Pinecone index to use
            model_name: Name of the Gemini model for answer generation
            use_gemini_embeddings: Whether to use Gemini for embeddings (True) or sentence-transformers (False)
            embedding_model_name: Name of the sentence-transformer model to use
        """
        # Ensure event loop is available
        ensure_event_loop()
        
        self.index_name = index_name
        self.model_name = model_name
        self.use_gemini_embeddings = use_gemini_embeddings
        self.embedding_model_name = embedding_model_name
        self.is_initialized = False
        self.index_dimension = None
        
        # Initialize Pinecone first to get index dimension
        if not PINECONE_API_KEY:
            print("Warning: PINECONE_API_KEY environment variable not set. Pinecone features will be disabled.")
            return
        
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists
            existing_indexes = pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                print(f"Warning: Pinecone index '{self.index_name}' not found.")
                if self.use_gemini_embeddings:
                    print("Did you run the create_gemini_embeddings.py script to create the Gemini embeddings?")
                else:
                    print("Did you run the create_pinecone_embeddings.py script to create standard embeddings?")
                self.is_initialized = False
                return
            
            # Connect to the index
            self.index = pc.Index(self.index_name)
            
            # Get index details to determine dimension
            index_stats = self.index.describe_index_stats()
            if 'dimension' in index_stats:
                self.index_dimension = index_stats['dimension']
            else:
                print(f"Warning: Could not determine dimension for index '{self.index_name}'")
                self.is_initialized = False
                return
                
            print(f"Successfully connected to Pinecone index '{self.index_name}' with dimension {self.index_dimension}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.is_initialized = False
            return
        
        # Initialize embedding models based on choice and index dimension
        if self.use_gemini_embeddings:
            if not GOOGLE_API_KEY:
                print("Warning: GOOGLE_API_KEY environment variable not set. Gemini embeddings cannot be used.")
                return
                
            try:
                # Configure Gemini
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_embedding_model_name = "models/embedding-001"
                
                # Check if Gemini embedding dimension matches index dimension
                # Gemini embedding-001 produces 768-dimensional vectors
                if self.index_dimension != 768:
                    print(f"Warning: Gemini embeddings (768 dimensions) don't match index dimension ({self.index_dimension})")
                    print("Switching to sentence-transformer model that matches index dimension")
                    self.use_gemini_embeddings = False
                else:
                    print(f"Using Gemini for embeddings with index '{self.index_name}'")
                    self.gemini_embeddings_enabled = True
            except Exception as e:
                print(f"Error setting up Gemini embeddings: {e}")
                self.gemini_embeddings_enabled = False
                self.use_gemini_embeddings = False
        
        # If not using Gemini embeddings, initialize sentence-transformers
        if not self.use_gemini_embeddings:
            try:
                # Select appropriate model based on index dimension
                if self.index_dimension == 384:
                    # paraphrase-MiniLM-L6-v2 produces 384-dimensional vectors
                    self.embedding_model_name = "paraphrase-MiniLM-L6-v2"
                elif self.index_dimension == 768:
                    # Models that produce 768-dimensional vectors
                    self.embedding_model_name = "all-mpnet-base-v2"
                else:
                    print(f"Warning: No default model for dimension {self.index_dimension}, using provided model {self.embedding_model_name}")
                
                # Initialize the model
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
                # Verify model dimension
                sample_embedding = self.embedding_model.encode("test")
                model_dimension = len(sample_embedding)
                
                if model_dimension != self.index_dimension:
                    print(f"Error: Model {self.embedding_model_name} produces {model_dimension}-dimensional embeddings, " 
                          f"but index dimension is {self.index_dimension}")
                    self.is_initialized = False
                    return
                    
                print(f"Using sentence-transformer model '{self.embedding_model_name}' ({model_dimension} dimensions) "
                      f"for embeddings with index '{self.index_name}' ({self.index_dimension} dimensions)")
                self.gemini_embeddings_enabled = False
                self.is_initialized = True
            except Exception as e:
                print(f"Error initializing sentence transformer model: {e}")
                self.is_initialized = False
                return
        
        # Initialize Gemini for answer generation
        if not GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY environment variable not set. Gemini answer generation will be disabled.")
            self.gemini_enabled = False
        else:
            try:
                # Configure the Gemini API
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Load the Gemini model
                self.model = genai.GenerativeModel(model_name)
                self.gemini_enabled = True
                print(f"Gemini model '{model_name}' initialized successfully for answer generation.")
            except Exception as e:
                print(f"Error initializing Gemini model: {e}")
                self.gemini_enabled = False
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query from Pinecone."""
        if not self.is_initialized:
            return []
            
        try:
            # Generate embedding for the query based on the chosen method
            if self.use_gemini_embeddings and self.gemini_embeddings_enabled:
                # Use Gemini for query embedding
                try:
                    embedding_result = genai.embed_content(
                        model=self.gemini_embedding_model_name,
                        content=query,
                        task_type="retrieval_query"
                    )
                    query_embedding = embedding_result["embedding"]
                except Exception as e:
                    print(f"Error generating Gemini embedding for query: {e}")
                    return []
            else:
                # Use sentence-transformers for query embedding
                query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query the Pinecone index
            results = self.index.query(
                vector=query_embedding,
                top_k=min(n_results, 20),  # Get up to 20 results
                include_metadata=True
            )
            
            # Check if we have results
            if not results.matches:
                return []
                
            # Format the results
            formatted_results = []
            for match in results.matches:
                result = {
                    "content": match.metadata.get("content", ""),
                    "metadata": {
                        "title": match.metadata.get("title", "Untitled"),
                        "url": match.metadata.get("url", ""),
                        "source": match.metadata.get("source", "unknown")
                    },
                    "distance": 1 - match.score  # Convert similarity score to distance
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def _generate_gemini_answer(self, query: str, context: str) -> str:
        """Generate an answer using Gemini LLM."""
        if not self.gemini_enabled:
            return None
            
        try:
            # Create a prompt for Gemini
            prompt = f"""You are an Angel One customer support assistant. Use ONLY the following context to answer the question.
            If the answer isn't in the context, say "I don't have information about that in my knowledge base."
            
            CONTEXT:
            {context}
            
            QUESTION:
            {query}
            
            ANSWER:"""
            
            # Generate response with safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Generate response with safety settings and error handling
            try:
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if response and hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    print("Warning: Gemini response has unexpected format")
                    return None
            except Exception as content_e:
                print(f"Error generating content: {content_e}")
                
                # Try reinitializing the model
                try:
                    self.model = genai.GenerativeModel(self.model_name)
                    print(f"Reinitialized Gemini model '{self.model_name}'")
                except Exception as reinit_e:
                    print(f"Error reinitializing model: {reinit_e}")
                
                return None
                
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return None
    
    def _generate_basic_answer(self, query: str, context: str, docs: List[Dict]) -> str:
        """
        Generate an answer based on the query and context without using an LLM.
        This is a fallback method if Gemini is not available.
        """
        # Sort documents by distance (smaller distance = more relevant)
        sorted_docs = sorted(docs, key=lambda x: x["distance"])
        
        # Use more documents for a more comprehensive answer
        # Consider up to 3 most relevant documents
        top_docs = sorted_docs[:min(3, len(sorted_docs))]
        
        # Extract query keywords for better matching
        query_words = set(query.lower().split())
        
        # Add important keywords based on common questions
        important_keywords = {"how", "what", "where", "when", "why", "who", "which", "can", "do", "is", "are"}
        # Filter out less important words from query
        focused_query_words = {word for word in query_words if len(word) > 2 or word in important_keywords}
        
        # Collect answer parts from each top document
        all_answer_parts = []
        
        for doc in top_docs:
            content = doc["content"]
            sentences = content.split(". ")
            
            # If document is very short, include all of it
            if len(sentences) <= 3:
                all_answer_parts.extend(sentences)
                continue
                
            # Include first sentence for context
            if sentences[0] not in all_answer_parts:
                all_answer_parts.append(sentences[0])
            
            # Score each sentence based on keyword matches
            scored_sentences = []
            for sentence in sentences[1:]:
                sentence_lower = sentence.lower()
                # Count matches with query keywords
                score = sum(1 for word in focused_query_words if word in sentence_lower)
                scored_sentences.append((score, sentence))
            
            # Sort sentences by score (highest first) and add top ones
            scored_sentences.sort(reverse=True)
            for _, sentence in scored_sentences[:3]:  # Take up to 3 best sentences per document
                if sentence not in all_answer_parts:
                    all_answer_parts.append(sentence)
        
        # Limit the total number of sentences to avoid overly long answers
        if len(all_answer_parts) > 8:
            all_answer_parts = all_answer_parts[:8]
            
        # Join the selected sentences to form the answer
        answer = ". ".join(all_answer_parts)
        if not answer.endswith("."):
            answer += "."
            
        return answer
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a user query using Pinecone retrieval and Gemini for answer generation."""
        if not self.is_initialized:
            return {
                "answer": "I'm sorry, but my knowledge base hasn't been properly initialized. Please ensure the data processing pipeline has been run correctly.",
                "sources": [],
                "has_answer": False
            }
            
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return {
                "answer": "I don't know the answer to that question.",
                "sources": [],
                "has_answer": False
            }
        
        # Check if retrieved documents are relevant enough
        is_relevant = any(doc["distance"] < 0.4 for doc in retrieved_docs)
        
        if not is_relevant:
            return {
                "answer": "I don't know the answer to that question.",
                "sources": [],
                "has_answer": False
            }
        
        # Prepare context by joining retrieved document contents
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        
        # Prepare sources for citation
        sources = []
        for doc in retrieved_docs:
            if doc["metadata"]["url"] not in [s["url"] for s in sources]:
                sources.append({
                    "title": doc["metadata"]["title"],
                    "url": doc["metadata"]["url"]
                })
        
        # Try to generate answer with Gemini
        if self.gemini_enabled:
            gemini_answer = self._generate_gemini_answer(query, context)
            
            if gemini_answer:
                return {
                    "answer": gemini_answer,
                    "sources": sources,
                    "has_answer": True
                }
        
        # Fall back to basic answer generation if Gemini fails
        base_answer = self._generate_basic_answer(query, context, retrieved_docs)
        
        return {
            "answer": base_answer,
            "sources": sources,
            "has_answer": True
        } 