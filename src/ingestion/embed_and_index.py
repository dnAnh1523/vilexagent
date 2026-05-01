from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType
)
from src.utils.logger import logger
import torch
import uuid

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 1

EMBEDDING_CONFIG = {
    "domestic": {
        "model": "jinaai/jina-embeddings-v5-text-small",
        "collection": "vilexagent_domestic",
        "vector_size": 1024,
    },
    "international": {
        "model": "jinaai/jina-embeddings-v5-text-small",
        "collection": "vilexagent_international",
        "vector_size": 1024,
    }
}

def get_device():
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.warning("No GPU detected, using CPU")
    return "cpu"

def create_collection(client: QdrantClient, name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        logger.warning(f"Collection '{name}' exists — deleting and recreating")
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    for field in ["domain", "source", "language", "tinh_trang_hieu_luc", "loai_van_ban"]:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )

    logger.success(f"Collection '{name}' created (dim={vector_size})")

def index_chunks(
    df: pd.DataFrame,
    model: SentenceTransformer,
    client: QdrantClient,
    config: dict,
    source_label: str
):
    texts = df["text"].tolist()
    total = len(texts)
    total_indexed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        batch_rows = df.iloc[batch_start:batch_end]

        embeddings = model.encode(
            batch_texts,
            task="retrieval",
            prompt_name="document",
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        points = []
        for embedding, (_, row) in zip(embeddings, batch_rows.iterrows()):
            payload = {
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]),
                "article_number": str(row["article_number"]),
                "text": str(row["text"]),
                "title": str(row.get("title", "")),
                "so_ky_hieu": str(row.get("so_ky_hieu", "")),
                "loai_van_ban": str(row.get("loai_van_ban", "")),
                "ngay_ban_hanh": str(row.get("ngay_ban_hanh", "")),
                "ngay_co_hieu_luc": str(row.get("ngay_co_hieu_luc", "")),
                "ngay_het_hieu_luc": str(row.get("ngay_het_hieu_luc", "")),
                "tinh_trang_hieu_luc": str(row.get("tinh_trang_hieu_luc", "")),
                "co_quan_ban_hanh": str(row.get("co_quan_ban_hanh", "")),
                "domain": str(row.get("domain", "")),
                "source": str(row.get("source", "")),
                "language": str(row.get("language", "")),
                "embedding_model": config["model"],
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            ))

        client.upsert(collection_name=config["collection"], points=points)
        total_indexed += len(points)

        if batch_end % 5000 == 0 or batch_end == total:
            logger.info(f"[{source_label}] Indexed {total_indexed}/{total} chunks...")

    logger.success(f"[{source_label}] Done — {total_indexed} chunks in '{config['collection']}'")

def embed_and_index():
    logger.info("Loading parsed chunks...")
    df = pd.read_parquet(PROCESSED_DIR / "parsed_chunks.parquet")
    logger.info(f"Total chunks: {len(df)}")

    df_domestic = df[df["source"] == "domestic"]
    df_international = df[df["source"] == "international"]
    logger.info(f"Domestic chunks: {len(df_domestic)}")
    logger.info(f"International chunks: {len(df_international)}")

    device = get_device()
    client = QdrantClient(url="http://localhost:6333")

    # --- Domestic ---
    cfg = EMBEDDING_CONFIG["domestic"]
    create_collection(client, cfg["collection"], cfg["vector_size"])
    logger.info(f"Loading domestic model: {cfg['model']}")
    model = SentenceTransformer(
        cfg["model"],
        device=device,
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "default_task": "retrieval",
            "load_in_4bit": True,
        }
    )
    model.max_seq_length = 512
    index_chunks(df_domestic, model, client, cfg, "domestic")
    del model
    torch.cuda.empty_cache()
    logger.info("Freed GPU memory after domestic model")

    # --- International ---
    cfg = EMBEDDING_CONFIG["international"]
    create_collection(client, cfg["collection"], cfg["vector_size"])
    if len(df_international) > 0:
        logger.info(f"Loading international model: {cfg['model']}")
        model = SentenceTransformer(
            cfg["model"], 
            device=device, 
            trust_remote_code=True, 
            model_kwargs={
                "torch_dtype": torch.bfloat16, 
                "default_task": "retrieval", 
                "load_in_4bit": True
            }
        )
        model.max_seq_length = 512
        index_chunks(df_international, model, client, cfg, "international")
        del model
        torch.cuda.empty_cache()
    else:
        logger.warning("No international chunks yet — collection created empty, ready for EVFTA/CPTPP docs")

    # Final verification
    for source, cfg in EMBEDDING_CONFIG.items():
        count = client.count(collection_name=cfg["collection"]).count
        logger.success(f"'{cfg['collection']}': {count} vectors")

if __name__ == "__main__":
    embed_and_index()