from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.utils.logger import logger
import torch
import uuid

PROCESSED_DIR = Path("data/processed")
COLLECTION_NAME = "vilexagent_international"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"
BATCH_SIZE = 8

def embed_international():
    logger.info("Loading international chunks...")
    df = pd.read_parquet(PROCESSED_DIR / "international_chunks.parquet")
    logger.info(f"Total chunks: {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16, "default_task": "retrieval", "load_in_4bit": True},
        trust_remote_code=True
    )

    client = QdrantClient(url="http://localhost:6333")

    texts = df["text"].tolist()
    total = len(texts)
    total_indexed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        batch_rows = df.iloc[batch_start:batch_end]

        with torch.amp.autocast('cuda'):
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
                "agreement": str(row.get("agreement", "")),
                "embedding_model": MODEL_NAME,
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_indexed += len(points)
        torch.cuda.empty_cache()

    count = client.count(collection_name=COLLECTION_NAME).count
    logger.success(f"'{COLLECTION_NAME}': {count} vectors")

if __name__ == "__main__":
    embed_international()