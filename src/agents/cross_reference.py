# src/agents/cross_reference.py
import os
import json
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

CROSS_REFERENCE_PROMPT = """Bạn là chuyên gia pháp lý so sánh luật Việt Nam với các hiệp định thương mại quốc tế.

Nhiệm vụ: Phân tích và đối chiếu các quy định pháp luật Việt Nam với các tiêu chuẩn quốc tế (EVFTA, CPTPP) dựa trên các đoạn văn bản được cung cấp.

## Câu hỏi gốc:
{original_query}

## Quy định pháp luật Việt Nam (nội địa):
{domestic_context}

## Tiêu chuẩn quốc tế (EVFTA/CPTPP):
{international_context}

## Yêu cầu phân tích:
1. Tóm tắt ngắn gọn nội dung chính từ pháp luật Việt Nam liên quan đến câu hỏi
2. Tóm tắt ngắn gọn các yêu cầu từ hiệp định quốc tế liên quan
3. Đánh giá mức độ tương thích:
   - "aligned": pháp luật Việt Nam đáp ứng đầy đủ yêu cầu quốc tế
   - "conflict": có mâu thuẫn trực tiếp giữa luật Việt Nam và yêu cầu quốc tế
   - "gap": pháp luật Việt Nam chưa đáp ứng đầy đủ hoặc còn thiếu sót
   - "no_international": không có tiêu chuẩn quốc tế liên quan trong dữ liệu
4. Giải thích cụ thể lý do đánh giá
5. Liệt kê các văn bản đã hết hiệu lực được tìm thấy (nếu có)

Trả lời ONLY bằng JSON hợp lệ, không có text nào khác:
{{
  "domestic_summary": "...",
  "international_summary": "...",
  "alignment": "aligned|conflict|gap|no_international",
  "explanation": "...",
  "expired_docs": ["tên văn bản 1", "tên văn bản 2"]
}}"""

def format_chunks_for_context(chunks: list[dict], max_chars_per_chunk: int = 400) -> str:
    if not chunks:
        return "Không có dữ liệu."

    lines = []
    for i, chunk in enumerate(chunks):
        title = chunk.get("title", "")[:60]
        article = chunk.get("article_number", "N/A")
        text = chunk.get("text", "")[:max_chars_per_chunk]
        status = chunk.get("tinh_trang_hieu_luc", "")
        agreement = chunk.get("agreement", "")
        source_label = f"[{agreement}]" if agreement else "[Việt Nam]"
        expired_note = " ⚠️ HẾT HIỆU LỰC" if "hết hiệu lực" in status.lower() else ""

        lines.append(
            f"[{i+1}] {source_label} {title} — {article}{expired_note}\n{text}"
        )

    return "\n\n".join(lines)

def cross_reference_node(state: AgentState) -> dict:
    original_query = state["original_query"]
    domestic_chunks = state.get("domestic_chunks", [])
    international_chunks = state.get("international_chunks", [])

    # Skip if no international chunks — no cross-reference needed
    if not international_chunks:
        logger.info("Cross-Reference: no international chunks, skipping")
        return {
            "cross_reference": {
                "domestic_summary": "",
                "international_summary": "",
                "alignment": "no_international",
                "explanation": "Câu hỏi chỉ liên quan đến pháp luật nội địa Việt Nam.",
                "expired_docs": []
            }
        }

    logger.info(f"Cross-Reference: analyzing {len(domestic_chunks)} domestic + {len(international_chunks)} international chunks")

    domestic_context = format_chunks_for_context(domestic_chunks)
    international_context = format_chunks_for_context(international_chunks)

    prompt = CROSS_REFERENCE_PROMPT.format(
        original_query=original_query,
        domestic_context=domestic_context,
        international_context=international_context
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        logger.success(f"Cross-Reference: alignment = {result['alignment']}")
        logger.info(f"  Explanation: {result['explanation'][:120]}...")
        if result.get("expired_docs"):
            logger.warning(f"  Expired docs flagged: {result['expired_docs']}")

        return {"cross_reference": result}

    except Exception as e:
        logger.error(f"Cross-Reference failed: {e}")
        return {
            "cross_reference": {
                "domestic_summary": "Lỗi phân tích",
                "international_summary": "Lỗi phân tích",
                "alignment": "gap",
                "explanation": f"Lỗi hệ thống: {str(e)}",
                "expired_docs": []
            }
        }