# src/agents/graph.py
import os
from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentState
from src.agents.query_decomposer import query_decomposer_node
from src.agents.domestic_retriever import domestic_retriever_node
from src.agents.international_retriever import international_retriever_node
from src.agents.cross_reference import cross_reference_node
from src.agents.synthesizer import synthesizer_node
from src.utils.logger import logger

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_decomposer", query_decomposer_node)
    graph.add_node("domestic_retriever", domestic_retriever_node)
    graph.add_node("international_retriever", international_retriever_node)
    graph.add_node("cross_reference", cross_reference_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.add_edge(START, "query_decomposer")

    # After decomposition, run both retrievers
    graph.add_edge("query_decomposer", "domestic_retriever")
    graph.add_edge("query_decomposer", "international_retriever")

    # After both retrievers complete, cross-reference
    graph.add_edge("domestic_retriever", "cross_reference")
    graph.add_edge("international_retriever", "cross_reference")

    # After cross-reference, synthesize
    graph.add_edge("cross_reference", "synthesizer")

    # End
    graph.add_edge("synthesizer", END)

    return graph.compile()

# Module-level compiled graph
vilexagent = build_graph()

def run_query(query: str) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Query: {query}")

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

    result = vilexagent.invoke(initial_state)

    logger.info(f"\n--- FINAL ANSWER ---")
    logger.info(result["final_answer"])
    logger.info(f"\n--- CITATIONS ---")
    for c in result["citations"]:
        logger.info(f"  {c}")
    logger.info(f"\nAlignment: {result['cross_reference']['alignment'] if result['cross_reference'] else 'N/A'}")
    logger.info(f"Expired docs: {result['has_expired_docs']}")

    return result

if __name__ == "__main__":
    # Test with all three query types
    queries = [
        # Type A — domestic only
        "Hợp đồng lao động phải có những nội dung gì theo quy định hiện hành?",
        # Type C — cross-corpus
        "Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?",
    ]

    for query in queries:
        run_query(query)
        print("\n" + "="*60 + "\n")