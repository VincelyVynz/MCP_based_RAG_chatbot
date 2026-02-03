from fastmcp import server
from fastmcp.tools import tool
from embeddings import load_vector_store, load_markdown, retrieve_top_k, MODEL_NAME
from sentence_transformers import SentenceTransformer

embeddings, chunks = load_vector_store()
model = SentenceTransformer(MODEL_NAME)

app = server.FastMCP(name = "RAG Chatbot with MCP")



if __name__ == "__main__":
    app.run()