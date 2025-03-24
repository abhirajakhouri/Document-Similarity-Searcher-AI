from fastapi import FastAPI, Form
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import psycopg2
import os

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and tokenizer for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_to_embeddings(texts):
    """Convert texts to embeddings using Hugging Face model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Connect to Supabase database using environment variables
conn = psycopg2.connect(
    database=os.environ.get("SUPABASE_DB"),  # Supabase database name
    user=os.environ.get("SUPABASE_USER"),    # Supabase username
    password=os.environ.get("SUPABASE_PASSWORD"),  # Supabase password
    host=os.environ.get("SUPABASE_HOST"),    # Supabase host URL
    port="5432",                             # Default PostgreSQL port
    sslmode="require"                        # Enable SSL for secure connection
)

cur = conn.cursor()

# Fetch documents from Supabase database during initialization
cur.execute("SELECT text FROM documents")  # Assuming 'text' column contains combined 'headline' and 'short_description'
documents = [row[0] for row in cur.fetchall()]
print(f"Fetched {len(documents)} documents from Supabase.")

# Create FAISS index and add embeddings (default: L2 metric)
index = faiss.IndexFlatL2(384)  # Assuming embedding size is 384 (adjust based on your embedding model)
embeddings = convert_to_embeddings(documents)
index.add(embeddings)
print("FAISS index created and embeddings added.")

class Query(BaseModel):
    query: str

@app.post("/api/search")
async def search(
    query: str = Form(...), 
    similarity_metric: str = Form(default="L2")
):
    """Search for similar documents based on the specified similarity metric."""
    
    # Handle similarity metric selection
    if similarity_metric == "L2":
        index_metric = faiss.IndexFlatL2(384)  # L2 distance metric (default)
    elif similarity_metric == "cosine":
        index_metric = faiss.IndexFlatIP(384)  # Cosine similarity metric (inner product)
    else:
        return {"error": f"Unsupported similarity metric: {similarity_metric}"}
    
    # Convert query to embedding and perform search
    query_embedding = convert_to_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]), k=5)  # Top 5 similar documents
    
    similar_documents = [documents[i] for i in I[0]]
    
    return {
        "similar_documents": similar_documents,
        "similarity_metric": similarity_metric,
        "distances": D.tolist()
    }

@app.post("/api/add_document")
async def add_document(document: str):
    """Add a new document to FAISS index and Supabase database."""
    
    new_embedding = convert_to_embeddings([document])[0]
    index.add(np.array([new_embedding]))
    
    # Add document to Supabase database
    cur.execute("INSERT INTO documents (text) VALUES (%s)", (document,))
    conn.commit()
    
    return {"message": "Document added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
