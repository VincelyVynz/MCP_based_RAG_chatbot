import os
from google.generativeai import Client as GeminiClient

import dotenv
dotenv.load_dotenv()

gemini = GeminiClient(api_key = os.getenv("GEMINI_API_KEY"))

def generate_response(prompt: str, model: str = "gemini-1.5") -> str:
    response = gemini.text.generate(model = model, prompt = prompt)
    return response.result[0].content