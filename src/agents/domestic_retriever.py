# src/agents/domestic_retriever.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.agents.state import AgentState
from src.utils.logger import logger
from src.utils.model_loader import get_embedding_model # Dùng model dùng chung
from dotenv import load_dotenv
load_dotenv()

COLLECTION = "vilexagent_domestic"
MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"
TOP_K = 5

# Module-level singleton — load once, reuse across calls

_client = None

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _client

def retrieve_domestic(query: str, domain: str) -> list[dict]:
    model = get_embedding_model() # Gọi model từ Singleton
    client = get_client()

    query_vector = model.encode(
        query,
        task="retrieval", # Chỉ dùng retrieval cho Jina v5
        prompt_name="query", # Bắt buộc phải có để phân biệt câu hỏi
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    filters = [FieldCondition(key="source", match=MatchValue(value="domestic"))]
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
            "so_ky_hieu": r.payload.get("so_ky_hieu", ""),
            "article_number": r.payload.get("article_number", ""),
            "text": r.payload.get("text", "")[:500],
            "score": round(r.score, 4),
            "source": "domestic",
            "domain": r.payload.get("domain", ""),
            "tinh_trang_hieu_luc": r.payload.get("tinh_trang_hieu_luc", ""),
            "loai_van_ban": r.payload.get("loai_van_ban", ""),
        }
        for r in results
    ]

def domestic_retriever_node(state: AgentState) -> dict:
    sub_questions = state["sub_questions"]
    domestic_sqs = [
        sq for sq in sub_questions
        if sq["source"] in ("domestic", "both")
    ]

    if not domestic_sqs:
        logger.info("Domestic Retriever: no domestic sub-questions, skipping")
        return {"domestic_chunks": []}

    logger.info(f"Domestic Retriever: processing {len(domestic_sqs)} sub-question(s)")

    all_chunks = []
    seen_chunk_ids = set()

    for sq in domestic_sqs:
        chunks = retrieve_domestic(sq["question"], sq["domain"])
        for chunk in chunks:
            if chunk["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["chunk_id"])
                all_chunks.append(chunk)

    # Sort by score descending, keep top 10 across all sub-questions
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    all_chunks = all_chunks[:10]

    # Flag expired documents
    expired = [
        c["title"] for c in all_chunks
        if "hết hiệu lực" in c["tinh_trang_hieu_luc"].lower()
    ]
    has_expired = len(expired) > 0

    logger.success(f"Domestic Retriever: {len(all_chunks)} unique chunks retrieved")
    if has_expired:
        logger.warning(f"Expired documents found: {len(expired)}")

    return {
        "domestic_chunks": all_chunks,
        "has_expired_docs": has_expired,
    }