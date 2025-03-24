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

# Step 1: Load the Blog Authorship Corpus dataset
file_path = r"C:\IIT-K\IITK_Acads\Intern\Task\GIVA\archive\blogtext.csv"

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Extract the blog text column (assuming it's named 'text')
documents = data['text'].tolist()

print(f"Loaded {len(documents)} blog posts.")

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

# Convert documents to embeddings and add them to FAISS index
embeddings = convert_to_embeddings(documents)
index.add(embeddings)

print("FAISS index created and embeddings added.")

# Step 4: Connect to Supabase database (use environment variables for credentials)
conn = psycopg2.connect(
    database=os.environ.get("SUPABASE_DB"),
    user=os.environ.get("SUPABASE_USER"),
    password=os.environ.get("SUPABASE_PASSWORD"),
    host=os.environ.get("SUPABASE_HOST"),
    port=os.environ.get("SUPABASE_PORT")
)

cur = conn.cursor()

# Insert documents into Supabase database (if not already done)
for doc in documents:
    cur.execute("INSERT INTO documents (text) VALUES (%s)", (doc,))
conn.commit()
print("Documents inserted into Supabase database.")

# Step 5: Define API endpoints
class Query(BaseModel):
    query: str

@app.post("/api/search")
async def search(query: Query):
    """Search for similar documents."""
    query_embedding = convert_to_embeddings([query.query])[0]
    D, I = index.search(np.array([query_embedding]), k=5)
    similar_documents = [documents[i] for i in I[0]]
