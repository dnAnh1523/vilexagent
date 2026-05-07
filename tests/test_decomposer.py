# src/agents/test_decomposer.py
from src.agents.state import AgentState
from src.agents.query_decomposer import query_decomposer_node
from src.utils.logger import logger

test_queries = [
    "Hợp đồng lao động phải có những nội dung gì theo quy định hiện hành?",
    "Điều kiện an toàn thực phẩm trong sản xuất thực phẩm xuất khẩu sang EU theo EVFTA là gì?",
    "Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?",
]

for query in test_queries:
    logger.info(f"\n{'='*60}")
    initial_state: AgentState = {
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
    result = query_decomposer_node(initial_state)
    logger.info(f"requires_international: {result['requires_international']}")
    logger.info(f"sub_questions: {result['sub_questions']}")