import os
import google.generativeai as genai
from mcp import MCPClient

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

mcp_client = MCPClient(
    server_url = "http://localhost:3333"
)

model = genai.GenerativeModel(
    model_name= "gemini-1.5-flash",
    tools = mcp_client.tools()
)

while True:
    user_input = input("You: ")
    if user_input.lower() in  ("exit", "quit"):
        break

    response = model.generate_content(user_input)
    print("Chatbot: ", response.text)