import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import psycopg2
import os

# Initialize FastAPI app
app = FastAPI()

#Step 1: Load dataset containing documents - based on US newspaper cuts 
# Set environment variables for Supabase integration
os.environ["SUPABASE_DB"] = "postgres"  
os.environ["SUPABASE_USER"] = "postgres.bbbysypaukepyqanshng"     
os.environ["SUPABASE_PASSWORD"] = "07RqfRyXz3mWeMIv" 
os.environ["SUPABASE_HOST"] = "aws-0-ap-south-1.pooler.supabase.com"     

try:
    # Connect to Supabase database using environment variables
    conn = psycopg2.connect(
        database=os.environ.get("SUPABASE_DB"),  # Supabase database name
        user=os.environ.get("SUPABASE_USER"),    # Supabase username
        password=os.environ.get("SUPABASE_PASSWORD"),  # Supabase password
        host=os.environ.get("SUPABASE_HOST"),    # Supabase host URL
        port="5432",                             # Default PostgreSQL port
        sslmode="require"                        # Enable SSL for secure connection
    )
    print("Connection to Supabase database successful!")
    cur.execute("SELECT * FROM documents")  # documents is table name
    rows = cur.fetchall()
    
    # Fetch data from the table 
    cur = conn.cursor()
    cur.execute("SELECT text FROM documents") #text column of documents table is dataset
    rows = cur.fetchall()

# Step 2: Load pre-trained model and tokenizer for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_to_embeddings(texts):
    """Convert texts to embeddings using Hugging Face model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Step 3: Create FAISS index and add embeddings
index = faiss.IndexFlatL2(384)  # Assuming embedding size is 384


#### IMPORTANT ####
####Need to answer with multiple models
####

# Convert documents to embeddings and add them to FAISS index
embeddings = convert_to_embeddings(rows)
index.add(embeddings)

print("FAISS index created and embeddings added.")

# Step 5: Define API endpoints
class Query(BaseModel):
    query: str

@app.post("/api/search")
async def search(query: Query):
    """Search for similar documents."""
    query_embedding = convert_to_embeddings([query.query])[0]
    D, I = index.search(np.array([query_embedding]), k=5) #use of vector embedding based search 
    similar_documents = [rows[i] for i in I[0]]
