import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import numpy as np
from mcp import MCPServer, Tool

TOP_K = 4

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

embedding_model = "gemini-embedding-1"

DATA_PATH = Path("./data/employee_data.txt")

raw_text = DATA_PATH.read_text(encoding="utf-8")

employee_chunks = [
    chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()
]

# Print to confirm document loading
print(f"Loaded {len(employee_chunks)} employee chunks")

def embed_text(text):
    result = genai.embed_content(
        model = embedding_model,
        content = text
    )
    return np.array([e["embedding"] for e in result["embedding"]], dtype="float32")

employee_embeddings = embed_text(employee_chunks)

dimension = employee_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(employee_embeddings)

def retrieve_context(query, top_k = TOP_K):
    query_vector = embed_text(query)
    distances, indices = index.search(query_vector, top_k)
    results = [employee_chunks[i] for i in indices[0]]
    return {"documents": results}

retrieve_tool = Tool(
    name = "retrieve_context",
    description = "Return the most relevant employee records for a query",
    input_schema = {"query" : str},
    output_schema = {"documents" : list},
    func = retrieve_context
)

server = MCPServer(tools=[retrieve_tool])
server.start()