import os
import shutil
import stat
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from git import Repo
from dotenv import load_dotenv

# --- NEW IMPORTS FOR SPRINT 2 ---
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Load your Gemini API key from the .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from the .env file!")

# --- CONFIGURE GEMINI FOR LLAMAINDEX ---
# We are using Gemini 2.5 Flash for the brain, and the 001 model for embeddings
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)

app = FastAPI(title="Codebase RAG API")

REPO_DIR = "cloned_repo"
DB_DIR = "chroma_db"  # The folder where our vector database will live

class RepoRequest(BaseModel):
    github_url: str

def remove_readonly(func, path, excinfo):
    """Helper function to force delete read-only files on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repository(repo_url: str, target_dir: str):
    """Clones a GitHub repository to a local directory, safely wiping previous clones."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, onerror=remove_readonly)
    
    try:
        print(f"Cloning {repo_url} into {target_dir}...")
        Repo.clone_from(repo_url, target_dir)
        print("Cloning complete!")
        return True
    except Exception as e:
        print(f"Error cloning repo: {e}")
        return False

def build_vector_database(repo_path: str):
    """Reads the cloned code, embeds it slowly, and saves it to ChromaDB."""
    import time  # Importing here just to be safe
    print("Reading files from the repository...")
    
    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        recursive=True,
        required_exts=[".py", ".java", ".js", ".jsx", ".html", ".css", ".md", ".ts", ".tsx"],
        exclude=["**/node_modules/**", "**/venv/**", "**/.venv/**", "**/build/**", "**/dist/**", "**/.git/**"]
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} code files.")

    print("Initializing ChromaDB database...")
    db = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db.get_or_create_collection("codebase_collection")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # We initialize an empty index first
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print("Embedding code using Gemini (Throttled to respect free tier limits)...")
    # Manually insert each file one by one with a 3-second delay
    for i, doc in enumerate(documents):
        print(f"Processing file {i+1} of {len(documents)}...")
        index.insert(doc)
        time.sleep(3)  # 3-second pause prevents the 429 Quota Exceeded error
        
    print("Database build complete!")
    return len(documents)

@app.post("/ingest")
async def ingest_repo(request: RepoRequest):
    url = request.github_url
    
    if "github.com" not in url:
        raise HTTPException(status_code=400, detail="Please provide a valid GitHub URL.")
    
    # Step 1: Clone the code
    success = clone_repository(url, REPO_DIR)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clone the repository.")
    
    # Step 2: Build the vector database
    try:
        num_files_indexed = build_vector_database(REPO_DIR)
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build vector database: {e}")
    
    return {
        "status": "success",
        "message": f"Successfully cloned and indexed {url}.",
        "files_indexed": num_files_indexed,
        "database_location": f"./{DB_DIR}"
    }

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_codebase(request: QueryRequest):
    # 1. Make sure the database actually exists first
    if not os.path.exists(DB_DIR):
        raise HTTPException(status_code=400, detail="Database not found. Please run /ingest first.")
        
    try:
        # 2. Connect to our saved Chroma database
        db = chromadb.PersistentClient(path=DB_DIR)
        chroma_collection = db.get_collection("codebase_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 3. Load the index into LlamaIndex
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # 4. Create the query engine
        # similarity_top_k=5 means we retrieve the 5 most relevant code chunks to answer the question
        query_engine = index.as_query_engine(similarity_top_k=5)
        
        # 5. Ask the question!
        print(f"Querying codebase for: '{request.question}'")
        response = query_engine.query(request.question)
        
        # Extracting the file names where the LLM found the answer (Huge resume flex!)
        source_files = list(set([node.node.metadata.get('file_path', 'Unknown file') for node in response.source_nodes]))
        
        return {
            "question": request.question,
            "answer": str(response),
            "sources": source_files
        }
        
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying database: {e}")