import asyncio
from fastmcp.client import Client
from chatbot.llm import generate_response

SERVER_URL = "http://localhost:3333"

async def query_employee_docs(query: str):
    async with Client(SERVER_URL) as client:
        context = await client.call_tool(
            "retrieve_employee_docs",
            {"query": query, "k": 3}
        )

        return context  #temporary

async def update_employee(employee_name: str, field: str, new_value: str):
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "update_employee_info",
            {
                "employee_name" : employee_name,
                "field" : field,
                "new_value": new_value
            }
        )
        return result

async def answer_query(query: str):
    context = await query_employee_docs(query)
    prompt = f"""
Use the following employee information to answer the question.

Docs:
{context}

Question:
{query}

Answer concisely:
"""
    answer = generate_response(prompt)
    return answer