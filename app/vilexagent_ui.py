# app/vilexagent_ui.py
import sys
import chainlit as cl
from dotenv import load_dotenv
sys.path.append(r"E:\\vilexagent")

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    actions = [
        cl.Action(
            name="faq",
            payload={"query": "Thời gian thử việc tối đa theo pháp luật lao động Việt Nam là bao lâu?"},
            label="📋 Thời gian thử việc tối đa",
        ),
        cl.Action(
            name="faq",
            payload={"query": "Nếu người sử dụng lao động đơn phương chấm dứt hợp đồng trái pháp luật thì phải bồi thường những gì cho người lao động?"},
            label="💼 Bồi thường khi sa thải trái luật",
        ),
        cl.Action(
            name="faq",
            payload={"query": "Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?"},
            label="🌐 Tiêu chuẩn lao động CPTPP",
        ),
    ]
 
    await cl.Message(
        content=(
            "# ⚖️ ViLexAgent\n"
            "**Hệ thống hỏi đáp pháp lý Việt Nam**\n\n"
            "Tôi có thể giúp bạn tra cứu:\n"
            "- 📋 Luật lao động, hợp đồng lao động, tiền lương\n"
            "- 🍽️ An toàn thực phẩm, kiểm dịch, xuất khẩu\n"
            "- 🌐 Tiêu chuẩn quốc tế EVFTA, CPTPP\n\n"
            "*Hoặc chọn câu hỏi mẫu bên dưới:*"
        ),
        actions=actions,
    ).send()
 
    cl.user_session.set("history", [])
 
 
@cl.action_callback("faq")
async def on_faq(action: cl.Action):
    query = action.payload.get("query", "")
    await on_message(cl.Message(content=query))

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return
 
    state = {
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
 
    async with cl.Step(name="💭 Thought", type="run") as thought:
        thought.input = query
 
        # --- Step 1: Query Decomposition ---
        async with cl.Step(name="🔍 Phân tích câu hỏi", type="tool") as step1:
            step1.input = query
            from src.agents.query_decomposer import query_decomposer_node
            decomp_result = query_decomposer_node(state)
            state.update(decomp_result)
            sub_questions = state.get("sub_questions", [])
            requires_intl = state.get("requires_international", False)
 
            sub_q_text = "\n".join([
                f"- [{sq['source']}|{sq['domain']}] {sq['question']}"
                for sq in sub_questions
            ])
            step1.output = (
                f"Chia thành **{len(sub_questions)}** câu hỏi con:\n{sub_q_text}\n"
                f"Cần tra cứu quốc tế: {'✅' if requires_intl else '❌'}"
            )
 
        # --- Step 2: Domestic Retrieval ---
        async with cl.Step(name="📚 Tra cứu pháp luật Việt Nam", type="tool") as step2:
            step2.input = f"{len([sq for sq in sub_questions if sq['source'] in ('domestic','both')])} câu hỏi nội địa"
            from src.agents.domestic_retriever import domestic_retriever_node
            domestic_result = domestic_retriever_node(state)
            state.update(domestic_result)
            domestic_chunks = state.get("domestic_chunks", [])
 
            if domestic_chunks:
                docs_text = "\n".join([
                    f"- {c['title'][:60]} — {c['article_number']} "
                    f"({'⚠️ hết hiệu lực' if 'hết hiệu lực' in c.get('tinh_trang_hieu_luc','').lower() else '✅ còn hiệu lực'})"
                    for c in domestic_chunks[:5]
                ])
                step2.output = f"Tìm thấy **{len(domestic_chunks)}** văn bản:\n{docs_text}"
            else:
                step2.output = "Không tìm thấy văn bản nội địa liên quan."
 
        # --- Step 3: International Retrieval (if needed) ---
        if requires_intl:
            async with cl.Step(name="🌐 Tra cứu tiêu chuẩn quốc tế", type="tool") as step3:
                step3.input = f"{len([sq for sq in sub_questions if sq['source'] in ('international','both')])} câu hỏi quốc tế"
                from src.agents.international_retriever import international_retriever_node
                intl_result = international_retriever_node(state)
                state.update(intl_result)
                intl_chunks = state.get("international_chunks", [])
 
                if intl_chunks:
                    intl_text = "\n".join([
                        f"- [{c.get('agreement','')}] {c['title'][:60]} — {c['article_number']}"
                        for c in intl_chunks[:5]
                    ])
                    step3.output = f"Tìm thấy **{len(intl_chunks)}** điều khoản quốc tế:\n{intl_text}"
                else:
                    step3.output = "Không tìm thấy điều khoản quốc tế liên quan."
 
        # --- Step 4: Cross-Reference --- luôn chạy
        async with cl.Step(name="⚖️ Đối chiếu pháp luật", type="tool") as step4:
            step4.input = (
                f"{len(state.get('domestic_chunks', []))} văn bản nội địa × "
                f"{len(state.get('international_chunks', []))} điều khoản quốc tế"
            )
            from src.agents.cross_reference import cross_reference_node
            xref_result = cross_reference_node(state)
            state.update(xref_result)
            cross_ref = state.get("cross_reference") or {}
 
            alignment = cross_ref.get("alignment", "unknown")
            alignment_emoji = {
                "aligned": "✅ Phù hợp",
                "conflict": "❌ Mâu thuẫn",
                "gap": "⚠️ Còn khoảng cách",
                "no_international": "ℹ️ Không áp dụng"
            }.get(alignment, alignment)
 
            step4.output = (
                f"Kết quả đối chiếu: **{alignment_emoji}**\n\n"
                f"{cross_ref.get('explanation', '')[:300]}"
            )
 
        # --- Step 5: Synthesis ---
        async with cl.Step(name="✍️ Tổng hợp câu trả lời", type="tool") as step5:
            step5.input = "Tổng hợp từ tất cả nguồn"
            from src.agents.synthesizer import synthesizer_node
            synth_result = synthesizer_node(state)
            state.update(synth_result)
            step5.output = f"Đã tạo câu trả lời ({len(state.get('final_answer',''))} ký tự)"
 
        thought.output = (
            f"{len(state.get('domestic_chunks', []))} văn bản nội địa · "
            f"{len(state.get('international_chunks', []))} điều khoản quốc tế · "
            f"{len(state.get('final_answer',''))} ký tự"
        )
 
    # --- Final Answer ---
    final_answer = state.get("final_answer", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
    citations = state.get("citations", [])
    has_expired = state.get("has_expired_docs", False)
 
    footer = ""
    if citations:
        footer += "\n\n---\n**📌 Nguồn tham khảo:**\n"
        for c in citations[:8]:
            footer += f"- {c}\n"
 
    if has_expired:
        footer += (
            "\n\n> ⚠️ **Lưu ý:** Một số văn bản pháp luật trong kết quả tra cứu "
            "đã **hết hiệu lực**. Vui lòng chỉ sử dụng văn bản còn hiệu lực làm căn cứ pháp lý."
        )
 
    await cl.Message(content=final_answer + footer).send()