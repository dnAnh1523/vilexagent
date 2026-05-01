# src/retrieval/baseline.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger
import torch

DOMESTIC_COLLECTION = "vilexagent_domestic"
INTERNATIONAL_COLLECTION = "vilexagent_international"
DOMESTIC_MODEL = "jinaai/jina-embeddings-v5-text-small"
INTERNATIONAL_MODEL = "jinaai/jina-embeddings-v5-text-small"
TOP_K = 5

class BaselineRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model on {device}...")
        self.model = SentenceTransformer(
            DOMESTIC_MODEL,
            device=device,
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True
        )
        self.client = QdrantClient(url="http://localhost:6333")
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


def run_verification():
    retriever = BaselineRetriever()

    test_queries = [
        {
            "query": "hợp đồng lao động phải có những nội dung gì",
            "source": "domestic",
            "domain": "labor",
            "description": "Type A — Single domestic labor query (Vietnamese)"
        },
        {
            "query": "điều kiện an toàn vệ sinh thực phẩm trong sản xuất",
            "source": "domestic",
            "domain": "food_safety",
            "description": "Type A — Single domestic food safety query (Vietnamese)"
        },
        {
            "query": "CPTPP labour obligations freedom of association Vietnam",
            "source": "international",
            "domain": "labor",
            "description": "Type C — International labor query (English)"
        },
        {
            "query": "EVFTA sanitary phytosanitary measures food import requirements",
            "source": "international",
            "domain": "food_safety",
            "description": "Type C — International food safety query (English)"
        },
    ]

    for test in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {test['description']}")
        logger.info(f"Text: {test['query']}")
        logger.info(f"Source: {test['source']} | Domain: {test['domain']}")

        results = retriever.retrieve(
            query=test["query"],
            source=test["source"],
            domain=test["domain"]
        )

        if not results:
            logger.warning("No results returned!")
            continue

        for i, r in enumerate(results):
            logger.info(f"\n  [{i+1}] Score: {r['score']:.4f}")
            logger.info(f"       Title: {r['title'][:60]}")
            logger.info(f"       Article: {r['article_number']}")
            logger.info(f"       Status: {r['tinh_trang_hieu_luc']}")
            logger.info(f"       Text: {r['text'][:150]}...")

if __name__ == "__main__":
    run_verification()