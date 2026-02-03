from fastmcp import server
from fastmcp.tools import tool
from embeddings import load_vector_store, load_markdown, retrieve_top_k, MODEL_NAME
from sentence_transformers import SentenceTransformer

embeddings, chunks = load_vector_store()
model = SentenceTransformer(MODEL_NAME)

app = server.FastMCP(name = "RAG Chatbot with MCP")

@tool()
def search_employees(query: str, k: int = 3) -> str:
    """
    Search employee docs using semantic similarity
    :param query:
    :param k:
    :return:
    """
    results = retrieve_top_k(
        query =query,
        embeddings= embeddings,
        chunks= chunks,
        model= model,
        k= k
    )

    response = []
    for r in results:
        response.append(
            f"Employee: {r['metadata']['employee_name']}\n{r['text']}"
        )
    return "\n\n---\n\n".join(response)


if __name__ == "__main__":
    app.run()