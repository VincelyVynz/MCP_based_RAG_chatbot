from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("Docs")
VECTOR_STORE_PATH = Path("vector_store")
VECTOR_STORE_PATH.mkdir(exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'

def load_markdown(filename: str) -> str:
    path = DOCS_PATH/filename
    return path.read_text(encoding="utf-8")

def chunk_employees(markdown_text: str):
    chunks = []
    sections = markdown_text.split("\n##")
    for section in sections[1:]:
        lines = section.strip().splitlines()
        name = lines[0].strip()
        text = "## " + section.strip()
        chunks.append({
            "text": text,
            "metadata": {"employee_name": name, "source": "employees.md"}
        })
    return chunks

def build_vector_store(chunks):
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    data = {"embeddings": np.array(embeddings), "chunks": chunks}
    with open (VECTOR_STORE_PATH/"index.pkl", "wb") as f:
        pickle.dump(data, f)

def load_vector_store():
    with open(VECTOR_STORE_PATH/"index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["chunks"]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_top_k(query: str, embeddings, chunks, model, k=3):
    query_vec = model.encode([query])[0]
    scores = [cosine_similarity(query_vec, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]