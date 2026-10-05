"""
raw_call.py: una llamada al modelo con HTTP puro (requests), sin SDK.
Fases 3 y 4 de la sesión de la Week 2: ver qué viaja, y luego leerlo sin romperse.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()  # lee .env si existe (lo creas en la Fase 5)
USE_LOCAL = os.getenv("USE_LOCAL_LLM", "true").strip().lower() == "true"
if USE_LOCAL:
    BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    MODEL = os.getenv("MODEL", "llama-3.2-3b-instruct")
else:
    BASE_URL = "https://api.openai.com/v1"
    MODEL = os.getenv("MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")

def extract_text(data: dict) -> str:
    """Lee la respuesta de un cuerpo de Chat Completions sin confiar en su forma."""
    if "error" in data:
        error = data["error"]
        return f"[error de la API] {error.get('message') if isinstance(error, dict) else error}"
    choices = data.get("choices") or []
    if not choices:
        return "[la respuesta no trae choices]"
    content = choices[0].get("message", {}).get("content")
    if content is None:
        return f"[sin texto, finish_reason={choices[0].get('finish_reason')}]"
    return content

headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. GET: ¿qué modelos ofrece este servidor?
models = requests.get(f"{BASE_URL}/models", headers=headers, timeout=10)
print("GET /models ->", models.status_code)
if models.ok:
    print("modelos:", [model["id"] for model in models.json()["data"]][:3])

# 2. POST: las cuatro piezas de la Week 1 (endpoint, key, body, respuesta)
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are an assistant for Café Aurora's staff. Answer in one sentence."},
        {"role": "user", "content": "What is a token in a language model?"},
    ],
    "temperature": 0.7,
    "max_tokens": 60,
}
response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
print("POST /chat/completions ->", response.status_code)
data = response.json()
print("respuesta:", extract_text(data))
print("usage:", json.dumps(data.get("usage")))