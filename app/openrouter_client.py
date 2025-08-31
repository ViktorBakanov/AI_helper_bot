import os
import httpx
import re
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-3.3-70b-instruct")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "My App",
    "Content-Type": "application/json"
}

async def ask_openrouter(prompt: str):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    print("==== ОТПРАВКА ЗАПРОСА В OpenRouter ====")
    print(f"Модель: {MODEL_ID}")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return re.sub(r'[^\u0000-\u007F\u0400-\u04FF\s.,!?()\[\]\"\'\-–—:;]', '', response.json()["choices"][0]["message"]["content"])