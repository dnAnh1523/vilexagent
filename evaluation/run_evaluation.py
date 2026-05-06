# evaluation/run_evaluation.py
"""
VilexAgent Evaluation Framework — RAGAS-free
=============================================
Tier 1: Structural / Deterministic checks (no LLM needed)
Tier 2: Golden Dataset checks (keyword + label matching)
Tier 3: LLM-as-Judge using project's own LLM (sequential, no burst)
"""

import os
import json
import time
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import logger

load_dotenv()

# -------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------
REQUIRED_ENV_VARS = ["FREELLM_API_KEY", "FREELLM_BASE_URL"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

BENCHMARK_PATH = Path("evaluation/benchmark.json")
RESULTS_DIR = Path("evaluation/results/custom_eval_result")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PER_QUESTION_DELAY   = int(os.getenv("EVAL_QUESTION_DELAY", 5))
BETWEEN_SYSTEMS_DELAY = int(os.getenv("EVAL_SYSTEM_DELAY",  20))
LLM_JUDGE_DELAY      = int(os.getenv("EVAL_JUDGE_DELAY",    5))

VALID_SOURCES    = {"domestic", "international", "both"}
VALID_DOMAINS    = {"labor", "food_safety"}
VALID_ALIGNMENTS = {"aligned", "conflict", "gap", "no_international"}

# -------------------------------------------------------
# Singleton Managers
# -------------------------------------------------------
_baseline_retriever = None
_reranker           = None
_answer_llm         = None

def get_baseline_retriever():
    global _baseline_retriever
    if _baseline_retriever is None:
        from src.retrieval.baseline import BaselineRetriever
        _baseline_retriever = BaselineRetriever()
    return _baseline_retriever

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        import torch
        logger.info("Initializing reranker...")
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.success("Reranker ready")
    return _reranker

def get_answer_llm():
    global _answer_llm
    if _answer_llm is None:
        from src.utils.llm import get_llm
        _answer_llm = get_llm()
    return _answer_llm

# -------------------------------------------------------
# Data Loading
# -------------------------------------------------------
def load_benchmark() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} benchmark questions from {BENCHMARK_PATH}")
    return data

# -------------------------------------------------------
# Domain Detection (for Baselines that don't decompose)
# -------------------------------------------------------
LABOR_KEYWORDS = [
    "lao động", "hợp đồng", "tiền lương", "công đoàn",
    "thử việc", "làm thêm", "thôi việc", "sa thải",
    "bảo hiểm xã hội", "người lao động", "tranh chấp",
    "đình công", "thương lượng", "trợ cấp"
]

def detect_domain(query: str) -> str:
    q = query.lower().strip()
    return "labor" if any(w in q for w in LABOR_KEYWORDS) else "food_safety"

# -------------------------------------------------------
# System Implementations
# -------------------------------------------------------
def run_baseline1(query: str) -> tuple[str, list[str]]:
    retriever = get_baseline_retriever()
    llm       = get_answer_llm()
    domain    = detect_domain(query)
    chunks    = retriever.retrieve(query, source="domestic", domain=domain, top_k=5)
    context   = "\n\n".join([c["text"] for c in chunks]) if chunks else "Không có ngữ cảnh."
    prompt    = (
        f"Dựa trên thông tin pháp luật sau, hãy trả lời câu hỏi:\n\n"
        f"{context}\n\nCâu hỏi: {query}"
    )
    return llm.invoke(prompt).content, [c["text"] for c in chunks]

def run_baseline2(query: str) -> tuple[str, list[str]]:
    retriever = get_baseline_retriever()
    reranker  = get_reranker()
    llm       = get_answer_llm()
    domain    = detect_domain(query)
    chunks    = retriever.retrieve(query, source="domestic", domain=domain, top_k=10)
    try:
        pairs  = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        chunks = [c for _, c in ranked[:5]]
    except Exception as e:
        logger.warning(f"Reranker failed: {e}. Using top-5 unranked.")
        chunks = chunks[:5]
    context = "\n\n".join([c["text"] for c in chunks]) if chunks else "Không có ngữ cảnh."
    prompt  = (
        f"Dựa trên thông tin pháp luật sau, hãy trả lời câu hỏi:\n\n"
        f"{context}\n\nCâu hỏi: {query}"
    )
    return llm.invoke(prompt).content, [c["text"] for c in chunks]

def run_vilexagent(query: str) -> tuple[str, list[str], dict]:
    from src.agents.graph import vilexagent
    initial_state = {
        "original_query":       query,
        "sub_questions":        [],
        "requires_international": False,
        "domestic_chunks":      [],
        "international_chunks": [],
        "cross_reference":      None,
        "final_answer":         "",
        "citations":            [],
        "has_expired_docs":     False,
        "error":                None,
    }
    result      = vilexagent.invoke(initial_state)
    all_chunks  = result.get("domestic_chunks", []) + result.get("international_chunks", [])
    contexts    = [c["text"] for c in all_chunks[:10]]
    final_answer = result.get("final_answer", "")
    if not final_answer:
        logger.warning(f"ViLexAgent returned empty answer for: {query[:50]}...")
    return final_answer, contexts, result

# -------------------------------------------------------
# TIER 1 — Structural / Deterministic Checks
# ViLexAgent only. Checks internal pipeline correctness.
# No LLM needed. Pure schema + string validation.
# -------------------------------------------------------

def check_decomposition_schema(sub_questions: list[dict]) -> dict:
    """Every sub-question must have valid source, domain, and non-empty question."""
    issues = []
    if not isinstance(sub_questions, list):
        return {"passed": False, "issues": ["sub_questions is not a list"]}
    if len(sub_questions) == 0:
        return {"passed": False, "issues": ["sub_questions is empty"]}
    for i, sq in enumerate(sub_questions):
        if not isinstance(sq, dict):
            issues.append(f"sub_question[{i}] is not a dict")
            continue
        if sq.get("source") not in VALID_SOURCES:
            issues.append(f"sub_question[{i}] invalid source: '{sq.get('source')}'")
        if sq.get("domain") not in VALID_DOMAINS:
            issues.append(f"sub_question[{i}] invalid domain: '{sq.get('domain')}'")
        if not sq.get("question", "").strip():
            issues.append(f"sub_question[{i}] empty question text")
    return {"passed": len(issues) == 0, "issues": issues}

def check_alignment_schema(cross_reference: dict) -> dict:
    """cross_reference must have a valid alignment value and non-empty summary fields."""
    issues = []
    if not isinstance(cross_reference, dict):
        return {"passed": False, "issues": ["cross_reference is not a dict"]}
    if cross_reference.get("alignment") not in VALID_ALIGNMENTS:
        issues.append(f"invalid alignment: '{cross_reference.get('alignment')}'")
    for key in ["domestic_summary", "international_summary", "explanation"]:
        if not cross_reference.get(key, "").strip():
            issues.append(f"missing/empty field: '{key}'")
    return {"passed": len(issues) == 0, "issues": issues}

def check_citation_presence(final_answer: str, citations: list[str]) -> dict:
    """
    At least one citation's so_ky_hieu should appear in final_answer.
    Soft check: no citations generated is a warning, not a fail.
    """
    if not citations:
        return {"passed": True, "grounded": False, "note": "no citations generated"}
    answer_lower = final_answer.lower()
    # Each citation is "SO_KY_HIEU — article_number"; check first part
    found = [c for c in citations if c.split("—")[0].strip().lower() in answer_lower]
    return {
        "passed":           True,         # not a hard structural fail
        "grounded":         len(found) > 0,
        "found_count":      len(found),
        "total_citations":  len(citations),
    }

def check_expired_warning(has_expired_docs: bool, final_answer: str) -> dict:
    """If expired docs exist in retrieved chunks, answer MUST warn the user."""
    if not has_expired_docs:
        return {"passed": True, "applicable": False}
    warn_phrases = [
        "hết hiệu lực", "không còn hiệu lực", "đã bị thay thế",
        "cảnh báo", "lưu ý", "văn bản cũ", "đã hết hiệu lực"
    ]
    warned = any(p in final_answer.lower() for p in warn_phrases)
    return {"passed": warned, "applicable": True, "warned": warned}

def run_tier1(full_state: dict, final_answer: str) -> dict:
    sub_questions  = full_state.get("sub_questions", [])
    cross_reference = full_state.get("cross_reference") or {}
    citations       = full_state.get("citations", [])
    has_expired     = full_state.get("has_expired_docs", False)

    d = check_decomposition_schema(sub_questions)
    a = check_alignment_schema(cross_reference)
    c = check_citation_presence(final_answer, citations)
    e = check_expired_warning(has_expired, final_answer)

    all_issues = d["issues"] + a["issues"]
    if not e["passed"]:
        all_issues.append("expired docs not warned in answer")

    return {
        "tier1_passed":           d["passed"] and a["passed"] and e["passed"],
        "decomp_schema_ok":       d["passed"],
        "alignment_schema_ok":    a["passed"],
        "citation_grounded":      c.get("grounded", False),
        "expired_warning_ok":     e["passed"],
        "tier1_issues":           all_issues,
    }

# -------------------------------------------------------
# TIER 2 — Golden Dataset Checks
# All systems. Compares behavior against expected labels
# and measures factual keyword coverage vs reference answer.
# -------------------------------------------------------

def evaluate_source_routing(sub_questions: list[dict], expected_source: str):
    """
    Did the decomposer correctly decide whether international sources are needed?
    Returns float (1.0/0.0) or None if not applicable.
    """
    if not sub_questions:
        return None
    pred_needs_intl = any(sq.get("source") in ("international", "both") for sq in sub_questions)
    exp_needs_intl  = expected_source in ("international", "both")
    return 1.0 if pred_needs_intl == exp_needs_intl else 0.0

def evaluate_alignment_label(predicted: str, expected: str):
    """Exact match of alignment label. Only meaningful for type_c questions."""
    if not predicted or not expected:
        return None
    return 1.0 if predicted == expected else 0.0

def evaluate_keyword_coverage(final_answer: str, reference: str) -> float:
    """
    Lightweight factual coverage proxy.
    Extracts meaningful tokens (>=4 chars, non-stopword) from the reference,
    checks what fraction appear in the answer.
    Score: 0.0 to 1.0
    """
    STOPWORDS = {
        "theo", "của", "trong", "được", "phải", "các", "này",
        "một", "và", "hoặc", "với", "cho", "tại", "từ", "đến",
        "là", "có", "không", "khi", "nếu", "đã", "sẽ", "bởi",
        "những", "như", "thì", "về", "lên", "trên", "dưới",
        "cũng", "rằng", "vào", "bằng", "đây", "đó", "mọi"
    }
    if not reference or not final_answer:
        return 0.0
    ref_tokens = {
        t for t in reference.lower().split()
        if len(t) >= 4 and t not in STOPWORDS
    }
    if not ref_tokens:
        return 0.0
    answer_lower = final_answer.lower()
    found = sum(1 for t in ref_tokens if t in answer_lower)
    return round(found / len(ref_tokens), 4)

def run_tier2(
    final_answer:     str,
    reference:        str,
    expected_source:  str,
    expected_alignment: str,
    question_type:    str,
    sub_questions:    list[dict] = None,
    cross_reference:  dict       = None,
) -> dict:
    source_score    = evaluate_source_routing(sub_questions or [], expected_source)
    alignment_score = None
    if question_type == "type_c" and cross_reference:
        alignment_score = evaluate_alignment_label(
            cross_reference.get("alignment"), expected_alignment
        )
    keyword_score = evaluate_keyword_coverage(final_answer, reference)
    return {
        "source_routing_score": source_score,
        "alignment_label_score": alignment_score,
        "keyword_coverage_score": keyword_score,
    }

# -------------------------------------------------------
# TIER 3 — LLM-as-Judge
# Uses the project's own LLM. Called SEQUENTIALLY — one
# call per question. No parallel jobs, no burst → no timeout.
# -------------------------------------------------------

LLM_JUDGE_PROMPT = """Bạn là chuyên gia đánh giá câu trả lời pháp lý Việt Nam.

Chấm điểm câu trả lời theo 3 tiêu chí (mỗi tiêu chí 1–5):

**Câu hỏi:**
{query}

**Câu trả lời tham chiếu:**
{reference}

**Câu trả lời cần đánh giá:**
{answer}

**Tiêu chí:**
- relevance (1–5): Có trả lời đúng câu hỏi không? (1=hoàn toàn lạc đề, 5=chính xác đầy đủ)
- groundedness (1–5): Có trích dẫn điều khoản/số hiệu văn bản cụ thể không? (1=không có, 5=rõ ràng cụ thể)
- completeness (1–5): So với tham chiếu, có đề cập đủ các điểm chính không? (1=thiếu hầu hết, 5=đầy đủ)

Trả lời ONLY bằng JSON hợp lệ, không có text nào khác:
{{"relevance": N, "groundedness": N, "completeness": N}}"""

def run_tier3_judge(query: str, reference: str, answer: str) -> dict:
    if not answer or not answer.strip():
        return {"relevance": 1, "groundedness": 1, "completeness": 1, "judge_error": "empty answer"}

    llm    = get_answer_llm()
    prompt = LLM_JUDGE_PROMPT.format(
        query=query,
        reference=reference,
        answer=answer[:1500],  # prevent context overflow
    )
    for attempt in range(2):
        try:
            raw = llm.invoke(prompt).content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw    = raw.strip()
            scores = json.loads(raw)
            for key in ["relevance", "groundedness", "completeness"]:
                if key not in scores:
                    raise ValueError(f"Missing key: {key}")
                scores[key] = max(1, min(5, int(scores[key])))  # clamp to 1-5
            scores["judge_error"] = None
            return scores
        except Exception as e:
            logger.warning(f"LLM Judge attempt {attempt+1} failed: {e}")
            time.sleep(3)

    return {
        "relevance": None, "groundedness": None, "completeness": None,
        "judge_error": "failed after 2 attempts"
    }

# -------------------------------------------------------
# Main Evaluation Runner
# -------------------------------------------------------

def run_evaluation(
    system_name:   str,
    benchmark:     list[dict],
    max_questions: int = None,
) -> tuple[pd.DataFrame, dict]:

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {system_name}")
    questions      = benchmark[:max_questions] if max_questions else benchmark
    is_vilexagent  = system_name == "ViLexAgent"
    raw_results    = []

    for i, q in enumerate(questions):
        logger.info(f"  [{i+1}/{len(questions)}] {q['id']}: {q['query'][:60]}...")
        start      = time.time()
        answer     = ""
        contexts   = []
        full_state = {}
        error_msg  = None

        # --- Run system ---
        try:
            if is_vilexagent:
                answer, contexts, full_state = run_vilexagent(q["query"])
            elif system_name == "Baseline1_NaiveRAG":
                answer, contexts = run_baseline1(q["query"])
            else:
                answer, contexts = run_baseline2(q["query"])
            latency = round(time.time() - start, 2)
            logger.info(f"    ✓ {latency}s")
        except Exception as e:
            error_msg = str(e)
            latency   = round(time.time() - start, 2)
            logger.error(f"    ✗ {error_msg}")

        # --- Tier 1 ---
        tier1 = {}
        if is_vilexagent and not error_msg:
            tier1 = run_tier1(full_state, answer)
            if not tier1["tier1_passed"]:
                logger.warning(f"    ⚠ T1 issues: {tier1['tier1_issues']}")

        # --- Tier 2 ---
        tier2 = {}
        if not error_msg:
            tier2 = run_tier2(
                final_answer       = answer,
                reference          = q["reference"],
                expected_source    = q["expected_source"],
                expected_alignment = q["expected_alignment"],
                question_type      = q["type"],
                sub_questions      = full_state.get("sub_questions", []) if is_vilexagent else [],
                cross_reference    = full_state.get("cross_reference")   if is_vilexagent else None,
            )

        # --- Tier 3 (throttled) ---
        tier3 = {}
        if not error_msg and answer.strip():
            time.sleep(LLM_JUDGE_DELAY)
            tier3 = run_tier3_judge(q["query"], q["reference"], answer)
            if tier3.get("judge_error"):
                logger.warning(f"    ⚠ Judge: {tier3['judge_error']}")
            else:
                logger.info(
                    f"    Judge → rel:{tier3['relevance']} "
                    f"grnd:{tier3['groundedness']} comp:{tier3['completeness']}"
                )

        raw_results.append({
            # Metadata
            "id":                  q["id"],
            "type":                q["type"],
            "domain":              q["domain"],
            "expected_source":     q["expected_source"],
            "expected_alignment":  q["expected_alignment"],
            "query":               q["query"],
            "answer":              answer,
            "reference":           q["reference"],
            "latency":             latency,
            "error":               error_msg,
            # Tier 1
            "t1_passed":              tier1.get("tier1_passed"),
            "t1_decomp_schema_ok":    tier1.get("decomp_schema_ok"),
            "t1_alignment_schema_ok": tier1.get("alignment_schema_ok"),
            "t1_citation_grounded":   tier1.get("citation_grounded"),
            "t1_expired_warning_ok":  tier1.get("expired_warning_ok"),
            # Tier 2
            "t2_source_routing":   tier2.get("source_routing_score"),
            "t2_alignment_label":  tier2.get("alignment_label_score"),
            "t2_keyword_coverage": tier2.get("keyword_coverage_score"),
            # Tier 3
            "t3_relevance":        tier3.get("relevance"),
            "t3_groundedness":     tier3.get("groundedness"),
            "t3_completeness":     tier3.get("completeness"),
        })

        time.sleep(PER_QUESTION_DELAY)

    # --- Build DataFrame & Summary ---
    df = pd.DataFrame(raw_results)

    def safe_mean(series):
        clean = series.dropna()
        return round(float(clean.mean()), 4) if len(clean) > 0 else None

    ok     = df[df["error"].isna()]
    type_c = ok[ok["type"] == "type_c"]

    def bool_rate(col):
        if is_vilexagent and col in ok.columns:
            return safe_mean(ok[col].map(lambda x: float(x) if x is not None else None))
        return None

    summary = {
        "system":            system_name,
        "total_questions":   len(questions),
        "error_rate":        round(float(df["error"].notna().mean()), 4),
        "avg_latency_s":     safe_mean(ok["latency"]),
        # Tier 1
        "t1_pass_rate":               bool_rate("t1_passed"),
        "t1_decomp_schema_rate":      bool_rate("t1_decomp_schema_ok"),
        "t1_citation_grounded_rate":  bool_rate("t1_citation_grounded"),
        "t1_expired_warning_rate":    bool_rate("t1_expired_warning_ok"),
        # Tier 2
        "t2_source_routing_acc": safe_mean(ok["t2_source_routing"])   if is_vilexagent else None,
        "t2_alignment_acc":      safe_mean(type_c["t2_alignment_label"]) if is_vilexagent else None,
        "t2_keyword_coverage":   safe_mean(ok["t2_keyword_coverage"]),
        # Tier 3
        "t3_relevance_avg":    safe_mean(ok["t3_relevance"]),
        "t3_groundedness_avg": safe_mean(ok["t3_groundedness"]),
        "t3_completeness_avg": safe_mean(ok["t3_completeness"]),
    }

    logger.info(f"\n--- {system_name} Summary ---")
    for k, v in summary.items():
        if v is not None:
            logger.info(f"  {k}: {v}")

    return df, summary

# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
def main():
    benchmark = load_benchmark()

    logger.info("Pre-initializing singletons...")
    get_baseline_retriever()
    get_reranker()
    get_answer_llm()

    MAX_Q     = int(os.getenv("EVAL_MAX_QUESTIONS", 30))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("vilexagent_evaluation")

    all_summaries = []
    systems       = ["Baseline1_NaiveRAG", "Baseline2_RAGRerank", "ViLexAgent"]

    for idx, system_name in enumerate(systems):
        logger.info(f"\n>>> [{idx+1}/{len(systems)}] {system_name}")
        with mlflow.start_run(run_name=f"{system_name}_{timestamp}"):
            df, summary = run_evaluation(system_name, benchmark, max_questions=MAX_Q)

            csv_path = RESULTS_DIR / f"{system_name}_{timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.success(f"Saved → {csv_path}")

            for k, v in summary.items():
                if v is not None and isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_param("system", system_name)
            mlflow.log_param("num_questions", MAX_Q)
            mlflow.log_param("eval_framework", "deterministic_llm_judge")
            mlflow.log_artifact(str(csv_path))

            all_summaries.append(summary)

        if idx < len(systems) - 1:
            logger.info(f"Cooling down {BETWEEN_SYSTEMS_DELAY}s...")
            time.sleep(BETWEEN_SYSTEMS_DELAY)

    # Save & print comparison
    df_compare   = pd.DataFrame(all_summaries)
    compare_path = RESULTS_DIR / f"comparison_{timestamp}.csv"
    df_compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "system", "error_rate", "avg_latency_s",
        "t2_keyword_coverage",
        "t3_relevance_avg", "t3_groundedness_avg", "t3_completeness_avg",
        "t2_source_routing_acc", "t2_alignment_acc", "t1_pass_rate",
    ]
    available = [c for c in display_cols if c in df_compare.columns]
    logger.info(f"\n{'='*60}\nFINAL COMPARISON TABLE")
    logger.info("\n" + df_compare[available].to_string(index=False))
    logger.success(f"Comparison saved → {compare_path}")


if __name__ == "__main__":
    main()
# import os
# import json
# import time
# import mlflow
# import pandas as pd
# from pathlib import Path
# from dotenv import load_dotenv
# from ragas.metrics import faithfulness, context_precision
# from ragas import evaluate as ragas_evaluate
# from datasets import Dataset as HFDataset
# from src.utils.logger import logger
# from ragas.run_config import RunConfig

# load_dotenv()

# # -------------------------------------------------------
# # Configuration & Constants
# # -------------------------------------------------------
# REQUIRED_ENV_VARS = ["FREELLM_API_KEY", "FREELLM_BASE_URL"]
# for var in REQUIRED_ENV_VARS:
#     if not os.getenv(var):
#         raise ValueError(f"Missing required environment variable: {var}")

# BENCHMARK_PATH = Path("evaluation/benchmark.json")
# RESULTS_DIR = Path("evaluation/results")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# PER_QUESTION_DELAY = int(os.getenv("EVAL_QUESTION_DELAY", 7))
# BETWEEN_SYSTEMS_DELAY = int(os.getenv("EVAL_SYSTEM_DELAY", 30))

# # -------------------------------------------------------
# # Singleton Managers
# # -------------------------------------------------------
# _baseline_retriever = None
# _reranker = None
# _answer_llm = None
# _ragas_llm = None

# def get_baseline_retriever():
#     global _baseline_retriever
#     if _baseline_retriever is None:
#         from src.retrieval.baseline import BaselineRetriever
#         _baseline_retriever = BaselineRetriever()
#     return _baseline_retriever

# def get_reranker():
#     global _reranker
#     if _reranker is None:
#         from sentence_transformers import CrossEncoder
#         import torch
#         logger.info("Initializing reranker...")
#         _reranker = CrossEncoder(
#             "BAAI/bge-reranker-v2-m3",
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
#     return _reranker

# def get_answer_llm():
#     global _answer_llm
#     if _answer_llm is None:
#         from src.utils.llm import get_llm
#         _answer_llm = get_llm()
#     return _answer_llm

# def get_ragas_llm():
#     global _ragas_llm
#     if _ragas_llm is None:
#         from langchain_openai import ChatOpenAI
#         from langchain_core.callbacks import BaseCallbackHandler
#         from ragas.llms import LangchainLLMWrapper

#         class TokenUsageLogger(BaseCallbackHandler):
#             def on_llm_end(self, response, **kwargs):
#                 for gen in response.generations:
#                     for g in gen:
#                         usage = getattr(g, "generation_info", {}) or {}
#                         token_usage = usage.get("token_usage") or usage.get("usage", {})
#                         if token_usage:
#                             logger.debug(
#                                 f"RAGAS LLM tokens — "
#                                 f"prompt: {token_usage.get('prompt_tokens', '?')}, "
#                                 f"completion: {token_usage.get('completion_tokens', '?')}, "
#                                 f"total: {token_usage.get('total_tokens', '?')}"
#                             )

#         _ragas_llm = LangchainLLMWrapper(ChatOpenAI(
#             model="gemini-2.5-flash",
#             base_url=os.getenv("FREELLM_BASE_URL", "http://localhost:3001/v1"),
#             api_key=os.getenv("FREELLM_API_KEY"),
#             temperature=0,
#             timeout=60,
#             max_retries=3,
#             max_tokens=4096,  # temporarily high to observe real usage
#             callbacks=[TokenUsageLogger()]
#         ))
#     return _ragas_llm

# def load_benchmark() -> list[dict]:
#     if not BENCHMARK_PATH.exists():
#         raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")
#     with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     logger.info(f"Loaded {len(data)} benchmark questions from {BENCHMARK_PATH}")
#     return data

# # -------------------------------------------------------
# # Domain Detection
# # -------------------------------------------------------
# LABOR_KEYWORDS = [
#     "lao động", "hợp đồng", "tiền lương", "công đoàn",
#     "thử việc", "làm thêm", "thôi việc", "sa thải",
#     "bảo hiểm xã hội", "người lao động"
# ]

# def detect_domain(query: str) -> str:
#     if any(w in query.lower().strip() for w in LABOR_KEYWORDS):
#         return "labor"
#     return "food_safety"

# # -------------------------------------------------------
# # System Implementations
# # -------------------------------------------------------
# def run_baseline1(query: str) -> tuple[str, list[str]]:
#     retriever = get_baseline_retriever()
#     llm = get_answer_llm()
#     domain = detect_domain(query)
#     chunks = retriever.retrieve(query, source="domestic", domain=domain, top_k=5)
#     context_text = "\n\n".join([c["text"] for c in chunks]) if chunks else "Không có ngữ cảnh."
#     prompt = f"Dựa trên thông tin sau, hãy trả lời câu hỏi:\n\n{context_text}\n\nCâu hỏi: {query}"
#     response = llm.invoke(prompt)
#     return response.content, [c["text"] for c in chunks]

# def run_baseline2(query: str) -> tuple[str, list[str]]:
#     retriever = get_baseline_retriever()
#     reranker = get_reranker()
#     llm = get_answer_llm()
#     domain = detect_domain(query)
#     chunks = retriever.retrieve(query, source="domestic", domain=domain, top_k=10)
#     try:
#         pairs = [(query, c["text"]) for c in chunks]
#         scores = reranker.predict(pairs)
#         ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
#         chunks = [c for _, c in ranked[:5]]
#     except Exception as e:
#         logger.warning(f"Reranker failed: {e}. Falling back to top-5 unranked.")
#         chunks = chunks[:5]
#     context_text = "\n\n".join([c["text"] for c in chunks]) if chunks else "Không có ngữ cảnh."
#     prompt = f"Dựa trên thông tin sau, hãy trả lời câu hỏi:\n\n{context_text}\n\nCâu hỏi: {query}"
#     response = llm.invoke(prompt)
#     return response.content, [c["text"] for c in chunks]

# def run_vilexagent(query: str) -> tuple[str, list[str], dict]:
#     from src.agents.graph import vilexagent
#     initial_state = {
#         "original_query": query,
#         "sub_questions": [],
#         "requires_international": False,
#         "domestic_chunks": [],
#         "international_chunks": [],
#         "cross_reference": None,
#         "final_answer": "",
#         "citations": [],
#         "has_expired_docs": False,
#         "error": None
#     }
#     result = vilexagent.invoke(initial_state)
#     all_chunks = result.get("domestic_chunks", []) + result.get("international_chunks", [])
#     contexts = [c["text"] for c in all_chunks[:10]]
#     final_answer = result.get("final_answer", "")
#     if not final_answer:
#         logger.warning(f"ViLexAgent returned empty answer for: {query[:50]}...")
#     return final_answer, contexts, result

# # -------------------------------------------------------
# # Tier 1: Component Evaluation (ViLexAgent only)
# # -------------------------------------------------------
# def evaluate_decomposition(sub_questions: list[dict], expected_source: str) -> float:
#     if not sub_questions:
#         return 0.0 if expected_source in ("international", "both") else 1.0
#     predicted_needs_intl = any(
#         sq.get("source") in ("international", "both") for sq in sub_questions
#     )
#     expected_needs_intl = expected_source in ("international", "both")
#     return 1.0 if predicted_needs_intl == expected_needs_intl else 0.0

# def evaluate_cross_reference(predicted_alignment: str, expected_alignment: str) -> float:
#     return 1.0 if predicted_alignment == expected_alignment else 0.0

# def evaluate_legal_currency(has_expired_docs: bool, final_answer: str) -> float:
#     if not has_expired_docs:
#         return 1.0
#     if not final_answer:
#         return 0.0
#     warned = any(phrase in final_answer.lower() for phrase in [
#         "hết hiệu lực", "không còn hiệu lực", "đã bị thay thế",
#         "cảnh báo", "lưu ý", "văn bản cũ"
#     ])
#     return 1.0 if warned else 0.0

# # -------------------------------------------------------
# # Main Evaluation Runner
# # -------------------------------------------------------
# def run_evaluation(system_name: str, benchmark: list[dict], max_questions: int = None):
#     logger.info(f"\n{'='*60}")
#     logger.info(f"Evaluating: {system_name}")
#     questions = benchmark[:max_questions] if max_questions else benchmark
#     is_vilexagent = system_name == "ViLexAgent"
#     raw_results = []
#     ragas_samples = []

#     # Step 1: Run all queries
#     for i, q in enumerate(questions):
#         logger.info(f"  [{i+1}/{len(questions)}] {q['id']}: {q['query'][:60]}...")
#         start = time.time()
#         answer = ""
#         contexts = []
#         decomp_score = crossref_score = currency_score = None
#         error_msg = None

#         try:
#             if is_vilexagent:
#                 answer, contexts, full_state = run_vilexagent(q["query"])
#                 sub_questions = full_state.get("sub_questions", [])
#                 cross_ref = full_state.get("cross_reference") or {}
#                 has_expired = full_state.get("has_expired_docs", False)
#                 decomp_score = evaluate_decomposition(sub_questions, q["expected_source"])
#                 crossref_score = evaluate_cross_reference(
#                     cross_ref.get("alignment", "no_international"),
#                     q["expected_alignment"]
#                 )
#                 currency_score = evaluate_legal_currency(has_expired, answer)
#             elif system_name == "Baseline1_NaiveRAG":
#                 answer, contexts = run_baseline1(q["query"])
#             else:
#                 answer, contexts = run_baseline2(q["query"])

#             latency = round(time.time() - start, 2)
#             raw_results.append({
#                 "id": q["id"], "type": q["type"], "domain": q["domain"],
#                 "expected_source": q["expected_source"],
#                 "expected_alignment": q["expected_alignment"],
#                 "query": q["query"], "answer": answer,
#                 "reference": q["reference"], "latency": latency,
#                 "decomp_score": decomp_score,
#                 "crossref_score": crossref_score,
#                 "currency_score": currency_score,
#                 "error": None
#             })

#             if answer and answer.strip() and contexts:
#                 ragas_samples.append({
#                     "question": q["query"],
#                     "answer": answer,
#                     "contexts": contexts,
#                     "ground_truth": q["reference"]
#                 })

#             logger.info(f"    ✓ {latency}s")

#         except Exception as e:
#             error_msg = str(e)
#             logger.error(f"    ✗ {error_msg}")
#             raw_results.append({
#                 "id": q["id"], "type": q["type"], "domain": q["domain"],
#                 "expected_source": q["expected_source"],
#                 "expected_alignment": q["expected_alignment"],
#                 "query": q["query"], "answer": "",
#                 "reference": q["reference"], "latency": 0,
#                 "decomp_score": None, "crossref_score": None,
#                 "currency_score": None, "error": error_msg
#             })

#         time.sleep(PER_QUESTION_DELAY)

#     # Step 2: RAGAS 0.2.15 evaluation
#     logger.info(f"\nRunning RAGAS on {len(ragas_samples)} valid samples...")
#     df_ragas = None

#     if ragas_samples:
#         try:
#             ragas_llm = get_ragas_llm()
#             faithfulness.llm = ragas_llm
#             # answer_relevancy.llm = ragas_llm
#             context_precision.llm = ragas_llm

#             ragas_data = {
#                 "question": [s["question"] for s in ragas_samples],
#                 "answer": [s["answer"] for s in ragas_samples],
#                 "contexts": [s["contexts"] for s in ragas_samples],
#                 "ground_truth": [s["ground_truth"] for s in ragas_samples],
#             }
#             ragas_dataset = HFDataset.from_dict(ragas_data)
#             result = ragas_evaluate(
#                 ragas_dataset,
#                 metrics=[faithfulness, context_precision],
#                 run_config=RunConfig(max_workers=1, timeout=60, max_retries=3)
#             )
#             df_ragas = result.to_pandas()
#             logger.success(f"RAGAS complete: {result}")
#         except Exception as e:
#             logger.error(f"RAGAS failed: {e}")

#     # Step 3: Build dataframe
#     df_raw = pd.DataFrame(raw_results)
#     for col in ["faithfulness", "context_precision"]:
#         df_raw[col] = None

#     if df_ragas is not None:
#         valid_mask = (
#             (df_raw["error"].isna()) &
#             (df_raw["answer"].notna()) &
#             (df_raw["answer"].str.strip().astype(bool))
#         )
#         valid_indices = df_raw[valid_mask].index.tolist()
#         for j, idx in enumerate(valid_indices):
#             if j < len(df_ragas):
#                 for col in ["faithfulness", "context_precision"]:
#                     if col in df_ragas.columns:
#                         df_raw.loc[idx, col] = df_ragas[col].iloc[j]

#     # Step 4: Summary
#     def safe_mean(series):
#         clean = series.dropna()
#         return round(float(clean.mean()), 4) if len(clean) > 0 else None

#     summary = {
#         "system": system_name,
#         "total": len(questions),
#         "error_rate": round(df_raw["error"].notna().mean(), 4),
#         "avg_latency": safe_mean(df_raw[df_raw["error"].isna()]["latency"]),
#         "faithfulness": safe_mean(df_raw["faithfulness"]),
#         "context_precision": safe_mean(df_raw["context_precision"]),
#         "decomp_accuracy": safe_mean(df_raw["decomp_score"]) if is_vilexagent else None,
#         "crossref_accuracy": (
#             safe_mean(df_raw[df_raw["type"] == "type_c"]["crossref_score"])
#             if is_vilexagent else None
#         ),
#         "legal_currency_score": safe_mean(df_raw["currency_score"]) if is_vilexagent else None,
#     }

#     logger.info(f"\n--- {system_name} Summary ---")
#     for k, v in summary.items():
#         logger.info(f"  {k}: {v}")

#     return df_raw, summary

# # -------------------------------------------------------
# # Main Entry Point
# # -------------------------------------------------------
# def main():
#     benchmark = load_benchmark()
#     logger.info("Pre-initializing singletons...")
#     get_baseline_retriever()
#     get_reranker()
#     get_answer_llm()

#     MAX_Q = int(os.getenv("EVAL_MAX_QUESTIONS", 30))
#     timestamp = time.strftime("%Y%m%d_%H%M%S")

#     mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
#     mlflow.set_experiment("vilexagent_evaluation")

#     all_summaries = []
#     systems = ["Baseline1_NaiveRAG", "Baseline2_RAGRerank", "ViLexAgent"]

#     for system_name in systems:
#         logger.info(f"\n>>> Starting: {system_name}")
#         with mlflow.start_run(run_name=f"{system_name}_{timestamp}"):
#             df, summary = run_evaluation(system_name, benchmark, max_questions=MAX_Q)

#             csv_path = RESULTS_DIR / f"{system_name}_{timestamp}.csv"
#             df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#             logger.success(f"Saved → {csv_path}")

#             for k, v in summary.items():
#                 if v is not None and not isinstance(v, str):
#                     mlflow.log_metric(k, v)
#             mlflow.log_param("system", system_name)
#             mlflow.log_param("num_questions", MAX_Q)
#             mlflow.log_artifact(str(csv_path))

#             all_summaries.append(summary)
#             if system_name != systems[-1]:
#                 logger.info(f"Cooling down {BETWEEN_SYSTEMS_DELAY}s...")
#                 time.sleep(BETWEEN_SYSTEMS_DELAY)

#     df_compare = pd.DataFrame(all_summaries)
#     compare_path = RESULTS_DIR / f"comparison_{timestamp}.csv"
#     df_compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

#     cols = [
#         "system", "faithfulness", "context_precision",
#         "decomp_accuracy", "crossref_accuracy", "legal_currency_score", "avg_latency"
#     ]
#     available_cols = [c for c in cols if c in df_compare.columns]
#     logger.info(f"\n{'='*60}\nCOMPARISON TABLE")
#     logger.info(df_compare[available_cols].to_string())
#     logger.success(f"Saved comparison → {compare_path}")

# if __name__ == "__main__":
#     main()
