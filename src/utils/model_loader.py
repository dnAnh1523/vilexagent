# src/utils/model_loader.py
import torch
import gc
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger

_GLOBAL_MODEL = None
MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"

def get_embedding_model():
    """Lazy loads the embedding model into VRAM."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Jina v5 Model on {device}...")
        
        # Using 4-bit quantization to minimize VRAM usage on RTX 3050
        _GLOBAL_MODEL = SentenceTransformer(
            MODEL_NAME,
            device=device,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16, 
                "load_in_4bit": True,
                "default_task": "retrieval"
            }
        )
        logger.success("Embedding model loaded successfully.")
    return _GLOBAL_MODEL

def release_embedding_model():
    """Explicitly releases the embedding model from VRAM."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is not None:
        logger.info("Releasing embedding model from VRAM...")
        del _GLOBAL_MODEL
        _GLOBAL_MODEL = None
        
        # Force garbage collection and empty CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.success("VRAM cleared.")
