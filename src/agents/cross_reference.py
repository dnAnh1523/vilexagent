# src/agents/cross_reference.py
from src.utils.llm import get_llm
from src.agents.state import AgentState
from src.utils.logger import logger
from src.utils.json_utils import extract_json_from_llm_output
from dotenv import load_dotenv
load_dotenv()

llm = get_llm()

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

# -------------------------------------------------------
# Placeholder constants — used for early exit and error states
# These are descriptive strings that pass T1 schema validation
# -------------------------------------------------------
_DOMESTIC_ONLY_PAYLOAD = {
    "domestic_summary": "N/A – truy vấn chỉ liên quan đến pháp luật nội địa Việt Nam.",
    "international_summary": "N/A – không có tiêu chuẩn quốc tế liên quan đến truy vấn này.",
    "alignment": "no_international",
    "explanation": "Câu hỏi chỉ liên quan đến pháp luật nội địa Việt Nam, không yêu cầu đối chiếu quốc tế.",
    "expired_docs": [],
}

_ERROR_PAYLOAD_TEMPLATE = {
    "domestic_summary": "Lỗi phân tích – không thể trích xuất nội dung nội địa.",
    "international_summary": "Lỗi phân tích – không thể trích xuất nội dung quốc tế.",
    "alignment": "gap",
    "explanation": "",   # filled at runtime with actual error message
    "expired_docs": [],
}


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

    # Early exit: no international chunks — return descriptive placeholders, not empty strings
    # FIX: changed "" → descriptive strings so T1 schema validator passes
    if not international_chunks:
        logger.info("Cross-Reference: no international chunks, skipping with placeholder payload")
        return {"cross_reference": _DOMESTIC_ONLY_PAYLOAD}

    logger.info(
        f"Cross-Reference: analyzing {len(domestic_chunks)} domestic "
        f"+ {len(international_chunks)} international chunks"
    )

    domestic_context = format_chunks_for_context(domestic_chunks)
    international_context = format_chunks_for_context(international_chunks)

    prompt = CROSS_REFERENCE_PROMPT.format(
        original_query=original_query,
        domestic_context=domestic_context,
        international_context=international_context,
    )

    last_error = None

    # FIX: added retry loop (was missing entirely — a single bad response crashed the node)
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()

            if not raw:
                logger.warning(f"Cross-Reference attempt {attempt + 1}: empty response, retrying...")
                last_error = "empty response from LLM"
                continue

            # Use shared robust extractor
            result = extract_json_from_llm_output(raw)

            logger.success(f"Cross-Reference: alignment = {result['alignment']}")
            logger.info(f"  Explanation: {result['explanation'][:120]}...")
            if result.get("expired_docs"):
                logger.warning(f"  Expired docs flagged: {result['expired_docs']}")

            return {"cross_reference": result}

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Cross-Reference attempt {attempt + 1} failed: {last_error}")

    # All 3 attempts exhausted — return descriptive error payload, not silent "gap"
    logger.error(f"Cross-Reference failed after 3 attempts: {last_error}")
    error_payload = dict(_ERROR_PAYLOAD_TEMPLATE)
    error_payload["explanation"] = f"Lỗi hệ thống sau 3 lần thử: {last_error}"
    return {"cross_reference": error_payload}