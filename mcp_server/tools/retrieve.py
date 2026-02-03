from fastmcp.tools import tool
from sentence_transformers import SentenceTransformer
from ..embeddings import load_vector_store, load_markdown, retrieve_top_k, MODEL_NAME

model = SentenceTransformer(MODEL_NAME)
embeddings, chunks = load_vector_store()

@tool(name = "retrieve_employee_docs")
def retrieve_employee_docs(query: str, k: int = 3) -> str:
    """
    Search employee docs using semantic similarity
    :param query:
    :param k:
    :return:
    """
    results = retrieve_top_k(
        query=query,
        embeddings=embeddings,
        chunks=chunks,
        model=model,
        k=k
    )

    response = []
    for r in results:
        response.append(
            f"Employee: {r['metadata']['employee_name']}\n{r['text']}"
        )
    return "\n\n---\n\n".join(response)
