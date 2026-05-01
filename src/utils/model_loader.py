# src/utils/model_loader.py
import torch
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger

_GLOBAL_MODEL = None
MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"

def get_embedding_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Jina v5 Model (Singleton) on {device}...")
        _GLOBAL_MODEL = SentenceTransformer(
            MODEL_NAME,
            device=device,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16, 
                "load_in_4bit": True,      # Cực kỳ quan trọng cho RTX 3050
                "default_task": "retrieval" # Jina v5 gộp chung query và passage
            }
        )
    return _GLOBAL_MODEL