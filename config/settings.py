"""Application settings loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "jarvis_fi.db"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "250"))
GDELT_REQUEST_DELAY_SECONDS = float(os.getenv("GDELT_REQUEST_DELAY_SECONDS", "2.0"))
GDELT_LANGUAGE = os.getenv("GDELT_LANGUAGE", "English")
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "365"))
BACKFILL_QUERY_DAYS = int(os.getenv("BACKFILL_QUERY_DAYS", "7"))
SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "UTC")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", str(DATA_DIR / "chroma"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "jarvis_fi_articles")

ALPHAVANTAGE_API_KEY = os.getenv("AlphaVantage_API_KEY", os.getenv("ALPHAVANTAGE_API_KEY", ""))
ALPHAVANTAGE_RATE_DELAY_SECONDS = float(os.getenv("ALPHAVANTAGE_RATE_DELAY_SECONDS", "15"))
# Free tier only allows outputsize=compact (100 pts) for TIME_SERIES_DAILY.
# Set to "full" once on a premium plan for 20+ years of equity/ETF history.
ALPHAVANTAGE_OUTPUTSIZE = os.getenv("ALPHAVANTAGE_OUTPUTSIZE", "compact")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retires models periodically — verify with `python tests/llm.py` or the
# models list (https://console.groq.com/docs/models) and override in .env.
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GROQ_LARGE_MODEL = os.getenv("GROQ_LARGE_MODEL", "openai/gpt-oss-120b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "phi3:mini")
OLLAMA_LARGE_MODEL = os.getenv("OLLAMA_LARGE_MODEL", "mistral:7b-instruct")
