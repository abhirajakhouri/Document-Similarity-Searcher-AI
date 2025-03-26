import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np
import psycopg2
import os

# Initialize FastAPI app
app = FastAPI()

# Step 1: Connect to Supabase and fetch the dataset (text column of documents table)
os.environ["SUPABASE_DB"] = "postgres"  
os.environ["SUPABASE_USER"] = "postgres.bbbysypaukepyqanshng"     
os.environ["SUPABASE_PASSWORD"] = "07RqfRyXz3mWeMIv" 
os.environ["SUPABASE_HOST"] = "aws-0-ap-south-1.pooler.supabase.com"

rows = []  # Initialize rows to avoid issues if connection fails

try:
    # Connect to Supabase database using environment variables
    conn = psycopg2.connect(
        database=os.environ.get("SUPABASE_DB"),
        user=os.environ.get("SUPABASE_USER"),
        password=os.environ.get("SUPABASE_PASSWORD"),
        host=os.environ.get("SUPABASE_HOST"),
        port="5432",
        sslmode="require"
    )
    print("Connection to Supabase database successful!")
    
    # Fetch data from the table
    cur = conn.cursor()
    cur.execute("SELECT text FROM documents")  # Fetch only the `text` column from the `documents` table
    rows = cur.fetchall()
    rows = [row[0] for row in rows]  # Extract text values into a list

except Exception as e:
    print(f"Error connecting to Supabase database: {e}")

finally:
    if conn:
        conn.close()
        print("Connection closed.")

# Step 2: Load pre-trained model and tokenizer for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_to_embeddings(texts):
    """Convert texts to embeddings using Hugging Face model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Step 3: Create FAISS index and add embeddings (default metric: L2 distance)
index = faiss.IndexFlatL2(384)  # Assuming embedding size is 384

if rows:  # Ensure rows are not empty before processing
    file_path=r"C:\IIT-K\IITK_Acads\Intern\Task\GIVA\embeddings.npy"
    embeddings = np.load(file_path)
    index.add(embeddings)
    print("FAISS index created and embeddings added.")
else:
    print("No data available to create FAISS index.")

# Step 4: Define API endpoints
class Query(BaseModel):
    query: str

@app.post("/api/search")
async def search(query: Query):
    """Search for similar documents."""
    try:
        query_embedding = convert_to_embeddings([query.query])[0]
        D, I = index.search(np.array([query_embedding]), k=5)  # Top-5 similar documents based on vector search
        
        similar_documents = [rows[i] for i in I[0]]
        
        return {
            "similar_documents": similar_documents,
            "distances": D.tolist()
        }
    
    except Exception as e:
        return {"error": f"An error occurred during search: {e}"}
