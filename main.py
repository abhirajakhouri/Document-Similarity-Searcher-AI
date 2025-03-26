import gdown
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Step 1: Download embeddings.npy from Google Drive if not present locally
def download_embeddings(file_id, local_path):
    """Download embeddings.npy from Google Drive using gdown."""
    if not os.path.exists(local_path):
        print(f"Downloading embeddings from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        try:
            gdown.download(url, local_path, quiet=False)
            print(f"Embeddings saved to {local_path}.")
        except Exception as e:
            print(f"Failed to download embeddings: {e}")
    else:
        print(f"Embeddings already exist at {local_path}.")

# Google Drive File ID and local path for embeddings.npy
file_id = "1XnmuZkITZ6NNxPXnY3vvGogQcU6uttsQ"  # Extracted from the provided link
local_file_path = "embeddings.npy"

# Step 2: Download embeddings if not already present locally
download_embeddings(file_id, local_file_path)

# Step 3: Load embeddings from .npy file
try:
    embeddings = np.load(local_file_path)
    print(f"Loaded embeddings from '{local_file_path}'. Shape: {embeddings.shape}")
except FileNotFoundError:
    print(f"Error: '{local_file_path}' not found. Please ensure the file exists.")
    embeddings = None

# Step 4: Create FAISS index and add embeddings (default metric: L2 distance)
index = None

if embeddings is not None:
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Dimensionality of vectors
    index.add(embeddings)
    print("FAISS index created and embeddings added.")
else:
    print("No embeddings available to create FAISS index.")

# Step 5: Define API endpoints
class Query(BaseModel):
    query_embedding: list  # Accept query embedding as a list of floats

@app.post("/api/search")
async def search(query: Query):
    """Search for similar documents using precomputed embeddings."""
    if index is None:
        raise HTTPException(status_code=500, detail="FAISS index is not initialized.")
    
    try:
        # Convert query embedding to NumPy array
        query_embedding = np.array(query.query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Perform FAISS search
        D, I = index.search(query_embedding, k=5)  # Top-5 similar documents
        
        return {
            "indices": I.tolist(),  # Indices of similar documents
            "distances": D.tolist()  # Distances to similar documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {e}")
