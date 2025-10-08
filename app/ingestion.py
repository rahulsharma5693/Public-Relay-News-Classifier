# ingest_to_chroma.py
import psycopg2
import pandas as pd

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Step 1: Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="relevance_db",
    user="postgres",
    password=5693
)
# Step 2: Load examples
df = pd.read_sql("SELECT * FROM relevance_examples;", conn)
df['combined'] = "<outlet_name>" + df['outlet_name'] + "</outlet_name>" +  "<title>" + df['orig_title'] + "</title>" + "<body>" + df['orig_body'] + "</body>"
# Step 3: Convert to LangChain Documents
docs = []
for _, row in df.iterrows():
    docs.append(Document(
        page_content=row["combined"],
        metadata={
            "analysis_id": row["analysis_id"],
            "outlet_name": row["outlet_name"],
            "orig_title": row["orig_title"],
            "orig_body" : row["orig_body"],
            "relevance": row["relevance"]
        }
    ))

# Step 4: Initialize Chroma with persistence
embedding_model = OllamaEmbeddings(model="qwen3:1.7b")
vectorstore = Chroma(
    collection_name="relevance_examples",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# Step 5: Add documents (auto-persisted)
vectorstore.add_documents(docs)

print(f"✅ Ingested {len(docs)} documents into ChromaDB.")