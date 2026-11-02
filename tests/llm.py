"""Quick Groq connectivity smoke test.

Reads the API key from the environment (.env) — never hardcode secrets here.
Run:  python tests/llm.py
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not set — add it to your .env (see .env.example).")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model=os.getenv("GROQ_LARGE_MODEL", "openai/gpt-oss-120b"),
    messages=[{"role": "user", "content": "Name three geopolitical risks to oil markets right now."}],
    temperature=0.2,
)

print(response.choices[0].message.content)
