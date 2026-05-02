# src/agents/synthesizer.py
import os
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

SYNTHESIS_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật Việt Nam và các hiệp định thương mại quốc tế.

Dựa trên kết quả phân tích dưới đây, hãy trả lời câu hỏi của người dùng một cách rõ ràng, chính xác và có trích dẫn nguồn.

## Câu hỏi:
{original_query}

## Tóm tắt từ pháp luật Việt Nam:
{domestic_summary}

## Tóm tắt từ tiêu chuẩn quốc tế:
{international_summary}

## Đánh giá tương thích:
Mức độ: {alignment}
Giải thích: {explanation}

## Các văn bản pháp luật Việt Nam liên quan:
{domestic_citations}

## Các điều khoản quốc tế liên quan:
{international_citations}

{expired_warning}

## Yêu cầu trả lời:
1. Trả lời trực tiếp câu hỏi trong 1-2 câu đầu tiên
2. Giải thích chi tiết dựa trên các văn bản pháp luật
3. Nếu có sự khác biệt giữa luật Việt Nam và tiêu chuẩn quốc tế, nêu rõ khoảng cách đó
4. Trích dẫn cụ thể điều khoản, số hiệu văn bản
5. Nếu có văn bản hết hiệu lực, cảnh báo người dùng và chỉ dùng văn bản còn hiệu lực làm căn cứ chính

Trả lời bằng tiếng Việt, rõ ràng và chuyên nghiệp."""

def format_citations(chunks: list[dict]) -> str:
    if not chunks:
        return "Không có dữ liệu."
    lines = []
    for c in chunks[:5]:
        title = c.get("title", "")[:70]
        article = c.get("article_number", "N/A")
        so_ky_hieu = c.get("so_ky_hieu", "")
        status = c.get("tinh_trang_hieu_luc", "")
        agreement = c.get("agreement", "")
        expired = " ⚠️ HẾT HIỆU LỰC" if "hết hiệu lực" in status.lower() else ""
        ref = f"[{agreement}] " if agreement else ""
        lines.append(f"- {ref}{title} ({so_ky_hieu}) — {article}{expired}")
    return "\n".join(lines)

def synthesizer_node(state: AgentState) -> dict:
    original_query = state["original_query"]
    domestic_chunks = state.get("domestic_chunks", [])
    international_chunks = state.get("international_chunks", [])
    cross_reference = state.get("cross_reference", {})
    has_expired = state.get("has_expired_docs", False)

    logger.info("Synthesizer: generating final answer...")

    expired_warning = ""
    if has_expired:
        expired_warning = """## ⚠️ CẢNH BÁO:
Một số văn bản pháp luật được tìm thấy đã HẾT HIỆU LỰC. 
Hãy cảnh báo rõ ràng người dùng và chỉ sử dụng văn bản CÒN HIỆU LỰC làm căn cứ pháp lý chính."""

    prompt = SYNTHESIS_PROMPT.format(
        original_query=original_query,
        domestic_summary=cross_reference.get("domestic_summary", "Không có dữ liệu nội địa."),
        international_summary=cross_reference.get("international_summary", "Không áp dụng."),
        alignment=cross_reference.get("alignment", "unknown"),
        explanation=cross_reference.get("explanation", ""),
        domestic_citations=format_citations(domestic_chunks),
        international_citations=format_citations(international_chunks),
        expired_warning=expired_warning
    )

    try:
        response = llm.invoke(prompt)
        final_answer = response.content.strip()

        # Build citation list
        citations = []
        for c in domestic_chunks[:5]:
            so_ky_hieu = c.get("so_ky_hieu", "")
            article = c.get("article_number", "")
            if so_ky_hieu:
                citations.append(f"{so_ky_hieu} — {article}")

        for c in international_chunks[:5]:
            title = c.get("title", "")[:40]
            article = c.get("article_number", "")
            citations.append(f"{title} — {article}")

        logger.success(f"Synthesizer: answer generated ({len(final_answer)} chars)")

        return {
            "final_answer": final_answer,
            "citations": citations
        }

    except Exception as e:
        logger.error(f"Synthesizer failed: {e}")
        return {
            "final_answer": f"Lỗi hệ thống khi tổng hợp câu trả lời: {str(e)}",
            "citations": []
        }