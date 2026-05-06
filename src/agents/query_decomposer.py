# src/agents/query_decomposer.py
from src.utils.llm import get_llm
from src.agents.state import AgentState
from src.utils.logger import logger
from dotenv import load_dotenv
from src.utils.json_utils import extract_json_from_llm_output

load_dotenv()

llm = get_llm()

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

    last_error = None

    for attempt in range(3):
        try:
            response = llm.invoke(DECOMPOSER_PROMPT.format(query=query))
            raw = response.content.strip()

            if not raw:
                logger.warning(f"Attempt {attempt + 1}: empty response, retrying...")
                last_error = "empty response from LLM"
                continue

            # Use shared robust extractor instead of brittle split
            parsed = extract_json_from_llm_output(raw)

            sub_questions = parsed["sub_questions"]
            requires_international = parsed["requires_international"]

            logger.success(f"Decomposed into {len(sub_questions)} sub-question(s)")
            for sq in sub_questions:
                logger.info(f"  → [{sq['source']}|{sq['domain']}] {sq['question']}")

            return {
                "sub_questions": sub_questions,
                "requires_international": requires_international,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

    # All 3 attempts exhausted — safe fallback
    logger.error("Query Decomposer failed after 3 attempts, using safe fallback")
    return {
        "sub_questions": [
            {"question": query, "source": "domestic", "domain": "labor"}
        ],
        "requires_international": False,
        "error": last_error,
    }