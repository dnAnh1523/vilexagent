# src/agents/query_decomposer.py
import os
import json
from langgraph.graph import START
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

DECOMPOSER_PROMPT = """Bạn là một chuyên gia phân tích câu hỏi pháp lý Việt Nam.

Nhiệm vụ: phân tích câu hỏi và chia thành các câu hỏi con để tra cứu.

Với mỗi câu hỏi con, xác định:
- "question": nội dung câu hỏi con
- "source": nguồn cần tra cứu
  + "domestic": chỉ luật Việt Nam
  + "international": chỉ hiệp định quốc tế (EVFTA, CPTPP)
  + "both": cả hai
- "domain": lĩnh vực
  + "labor": lao động, việc làm, hợp đồng lao động, tiền lương, công đoàn
  + "food_safety": an toàn thực phẩm, vệ sinh thực phẩm, kiểm dịch

Trả lời ONLY bằng JSON hợp lệ, không có text nào khác:
{{
  "sub_questions": [
    {{
      "question": "...",
      "source": "domestic|international|both",
      "domain": "labor|food_safety"
    }}
  ],
  "requires_international": true|false
}}

Câu hỏi: {query}"""

def query_decomposer_node(state: AgentState) -> dict:
    query = state["original_query"]
    logger.info(f"Query Decomposer: '{query[:80]}'")

    try:
        response = llm.invoke(DECOMPOSER_PROMPT.format(query=query))
        raw = response.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        sub_questions = parsed["sub_questions"]
        requires_international = parsed["requires_international"]

        logger.success(f"Decomposed into {len(sub_questions)} sub-question(s)")
        for sq in sub_questions:
            logger.info(f"  → [{sq['source']}|{sq['domain']}] {sq['question']}")

        return {
            "sub_questions": sub_questions,
            "requires_international": requires_international,
            "error": None
        }

    except Exception as e:
        logger.error(f"Query Decomposer failed: {e}")
        return {
            "sub_questions": [{
                "question": query,
                "source": "domestic",
                "domain": "labor"
            }],
            "requires_international": False,
            "error": str(e)
        }