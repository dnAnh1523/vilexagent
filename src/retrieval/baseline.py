# src/retrieval/baseline.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.utils.model_loader import get_embedding_model
from src.utils.logger import logger

DOMESTIC_COLLECTION = "vilexagent_domestic"
INTERNATIONAL_COLLECTION = "vilexagent_international"
TOP_K = 5

class BaselineRetriever:
    def __init__(self):
        self.model = get_embedding_model()
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        logger.success("BaselineRetriever ready")

    def retrieve(self, query: str, source: str = "domestic", domain: str = None, top_k: int = TOP_K) -> list[dict]:
        collection = DOMESTIC_COLLECTION if source == "domestic" else INTERNATIONAL_COLLECTION

        query_vector = self.model.encode(
            query,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()

        filters = [FieldCondition(key="source", match=MatchValue(value=source))]
        if domain:
            filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=Filter(must=filters),
            limit=top_k,
            with_payload=True
        ).points

        return [
            {
                "score": r.score,
                "chunk_id": r.payload.get("chunk_id"),
                "title": r.payload.get("title"),
                "article_number": r.payload.get("article_number"),
                "text": r.payload.get("text", "")[:300],
                "tinh_trang_hieu_luc": r.payload.get("tinh_trang_hieu_luc"),
                "domain": r.payload.get("domain"),
                "source": r.payload.get("source"),
            }
            for r in results
        ]