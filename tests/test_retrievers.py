# src/agents/test_retrievers.py
from src.agents.state import AgentState
from src.agents.query_decomposer import query_decomposer_node
from src.agents.domestic_retriever import domestic_retriever_node
from src.agents.international_retriever import international_retriever_node
from src.utils.logger import logger

query = "Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?"

state: AgentState = {
    "original_query": query,
    "sub_questions": [],
    "requires_international": False,
    "domestic_chunks": [],
    "international_chunks": [],
    "cross_reference": None,
    "final_answer": "",
    "citations": [],
    "has_expired_docs": False,
    "error": None
}

logger.info("Step 1: Query Decomposer")
state.update(query_decomposer_node(state))

logger.info("\nStep 2: Domestic Retriever")
state.update(domestic_retriever_node(state))

logger.info("\nStep 3: International Retriever")
state.update(international_retriever_node(state))

logger.info(f"\n{'='*60}")
logger.info(f"Domestic chunks: {len(state['domestic_chunks'])}")
for c in state["domestic_chunks"][:3]:
    logger.info(f"  [{c['score']}] {c['title'][:50]} — {c['article_number']}")
    logger.info(f"    Status: {c['tinh_trang_hieu_luc']}")

logger.info(f"\nInternational chunks: {len(state['international_chunks'])}")
for c in state["international_chunks"][:3]:
    logger.info(f"  [{c['score']}] {c['title'][:50]} — {c['article_number']}")
    logger.info(f"    Agreement: {c['agreement']}")

logger.info(f"\nhas_expired_docs: {state['has_expired_docs']}")