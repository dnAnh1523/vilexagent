# src/agents/international_retriever.py
import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.agents.state import AgentState
from src.utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

COLLECTION = "vilexagent_international"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"
TOP_K = 5

_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            MODEL_NAME,
            device=device,
            model_kwargs={"dtype": torch.bfloat16},
            trust_remote_code=True
        )
    return _model

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _client

def retrieve_international(query: str, domain: str) -> list[dict]:
    model = get_model()
    client = get_client()

    query_vector = model.encode(
        query,
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    filters = [FieldCondition(key="source", match=MatchValue(value="international"))]
    if domain in ("labor", "food_safety"):
        filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(must=filters),
        limit=TOP_K,
        with_payload=True
    ).points

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "doc_id": r.payload.get("doc_id"),
            "title": r.payload.get("title", ""),
            "article_number": r.payload.get("article_number", ""),
            "text": r.payload.get("text", "")[:500],
            "score": round(r.score, 4),
            "source": "international",
            "domain": r.payload.get("domain", ""),
            "tinh_trang_hieu_luc": r.payload.get("tinh_trang_hieu_luc", ""),
            "agreement": r.payload.get("agreement", ""),
        }
        for r in results
    ]

def international_retriever_node(state: AgentState) -> dict:
    if not state.get("requires_international", False):
        logger.info("International Retriever: not required, skipping")
        return {"international_chunks": []}

    sub_questions = state["sub_questions"]
    intl_sqs = [
        sq for sq in sub_questions
        if sq["source"] in ("international", "both")
    ]

    if not intl_sqs:
        logger.info("International Retriever: no international sub-questions, skipping")
        return {"international_chunks": []}

    logger.info(f"International Retriever: processing {len(intl_sqs)} sub-question(s)")

    all_chunks = []
    seen_chunk_ids = set()

    for sq in intl_sqs:
        chunks = retrieve_international(sq["question"], sq["domain"])
        for chunk in chunks:
            if chunk["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["chunk_id"])
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    all_chunks = all_chunks[:10]

    logger.success(f"International Retriever: {len(all_chunks)} unique chunks retrieved")

    return {"international_chunks": all_chunks}