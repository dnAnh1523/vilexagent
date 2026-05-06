# src/utils/llm.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.utils.logger import logger

load_dotenv()

FREELLM_BASE_URL = os.getenv("FREELLM_BASE_URL", "http://localhost:3001/v1")
FREELLM_API_KEY = os.getenv("FREELLM_API_KEY")

if not FREELLM_API_KEY:
    raise ValueError("FREELLM_API_KEY not set in .env")

_llm = None

def get_llm(temperature: float = 0) -> ChatOpenAI:
    global _llm
    if _llm is None:
        logger.info(f"Initializing LLM via FreeLLMAPI at {FREELLM_BASE_URL}")
        _llm = ChatOpenAI(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            base_url=FREELLM_BASE_URL,
            api_key=FREELLM_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
        logger.success("LLM ready")
    return _llm