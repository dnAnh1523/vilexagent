import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from typing import TypedDict
from src.utils.logger import logger
import mlflow

load_dotenv()

# --- State definition ---
class AgentState(TypedDict):
    query: str
    response: str

# --- LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Nodes ---
def responder_node(state: AgentState) -> AgentState:
    logger.info(f"Responder node received query: {state['query']}")
    result = llm.invoke(state["query"])
    logger.info(f"LLM responded successfully")
    return {"response": result.content}

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("responder", responder_node)
graph.set_entry_point("responder")
graph.add_edge("responder", END)
app = graph.compile()

# --- Qdrant ---
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))

# --- MLflow ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# --- Run ---
if __name__ == "__main__":
    logger.info("Testing LangGraph...")
    result = app.invoke({"query": "What is RAG in one sentence?"})
    logger.success(f"LLM response: {result['response']}")

    logger.info("Testing Qdrant...")
    collections = qdrant.get_collections()
    logger.success(f"Qdrant collections: {collections}")

    logger.info("Testing MLflow...")
    with mlflow.start_run(run_name="verify-setup"):
        mlflow.log_param("phase", 0)
        mlflow.log_metric("setup_complete", 1.0)
    logger.success("MLflow run logged")

    logger.success("All systems operational. Phase 0 complete.")