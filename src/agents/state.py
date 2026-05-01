# src/agents/state.py
from typing import TypedDict, Optional, Annotated
import operator

class AgentState(TypedDict):
    # Input
    original_query: str

    # Query decomposition
    sub_questions: list[dict]      # each: {question, source, domain}
    requires_international: bool

    # Retrieval results — store only references, not full text
    domestic_chunks: Annotated[list[dict], operator.add]
    international_chunks: Annotated[list[dict], operator.add]

    # Cross-reference output
    cross_reference: Optional[dict]

    # Final output
    final_answer: str
    citations: list[str]
    has_expired_docs: bool

    # Control
    error: Optional[str]