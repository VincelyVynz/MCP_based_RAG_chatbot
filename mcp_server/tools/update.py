from fastmcp.tools import tool
from pathlib import Path
from sentence_transformers import SentenceTransformer
from ..embeddings import DOCS_PATH, build_vector_store, chunk_employees, load_markdown, MODEL_NAME

model = SentenceTransformer(MODEL_NAME)

@tool(name = "update_employee_info")
def update_employee_info(employee_name: str, field: str, new_value: str) -> str:
    """
    Update employee information in employee docs
    """
    path = DOCS_PATH/"employees.md"
    text = path.read_text(encoding="utf-8")
    sections = text.split("\n##")

    updated = False
    for i in range(1, len(sections)):
        if sections[i].startswith(employee_name):
            lines = sections[i].splitlines()
            for j, line in enumerate(lines):
                if line.lower().startswith(f"- {field.lower()}:"):
                  lines[j] = f"- {field}: {new_value}"
                  updated = True
                  break

            sections[i] = "\n".join(lines)

    if not updated:
        return f"Employee or field not found."
    updated_text = "\n##".join(sections)
    path.write_text(updated_text, encoding="utf-8")

    chunks = chunk_employees(updated_text)
    build_vector_store(chunks)
    return f"Updated {field} for {employee_name}"