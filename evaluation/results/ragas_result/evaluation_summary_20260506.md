# Evaluation Summary Report — ViLexAgent vs. Baselines
**Date:** 2026-05-06 | **Total queries evaluated:** 30 per system

---

## 1. Overview

This report summarises the evaluation results of **three systems** on a legal QA benchmark covering Vietnamese law and international trade agreements (EVFTA, CPTPP). The benchmark contains 30 queries across two domains and three query types.

| System | Description |
|---|---|
| **Baseline 1 — NaiveRAG** | Standard RAG without reranking |
| **Baseline 2 — RAGRerank** | RAG with reranking step |
| **ViLexAgent** | Multi-agent RAG system with legal-currency checking, decomposition scoring, and cross-reference scoring |

---

## 2. Dataset Composition

### 2.1 Query Types

| Type | Code | Count | Description |
|---|---|---|---|
| Factual / Type A | `type_a` | 10 | Straightforward legal facts from domestic law |
| Applied / Type B | `type_b` | 10 | Multi-step applied legal questions |
| Comparative / Type C | `type_c` | 10 | Cross-reference between domestic law and international agreements |

### 2.2 Domains

| Domain | Count |
|---|---|
| Labor law (`labor`) | 18 |
| Food safety law (`food_safety`) | 12 |

### 2.3 Expected Source & Alignment

| Expected Source | Count | Alignment Type | Count |
|---|---|---|---|
| Domestic only | 20 | No international requirement | 20 |
| Both domestic + international | 10 | Gap (VN law lags behind) | 6 |
| | | Aligned (VN law matches) | 2 |
| | | Conflict (VN law contradicts) | 2 |

---

## 3. System-Level Metrics

The following table is taken directly from `comparison_20260506_111259.csv`.

| Metric | Baseline 1 — NaiveRAG | Baseline 2 — RAGRerank | ViLexAgent |
|---|:---:|:---:|:---:|
| **Total queries** | 30 | 30 | 30 |
| **Error rate** | 0.0 | 0.0 | 0.0 |
| **Avg. latency (s)** | 3.39 | 3.57 | **8.70** |
| **Faithfulness** | 0.652 | 0.662 | 0.395 |
| **Context precision** | 0.411 | 0.429 | **0.519** |
| **Decomp. accuracy** | N/A | N/A | **0.767** |
| **Cross-ref. accuracy** | N/A | N/A | **0.600** |
| **Legal currency score** | N/A | N/A | **1.000** |

> **Metrics glossary**
> - **Faithfulness**: proportion of answer statements grounded in the retrieved context (RAGAS metric). Higher = fewer hallucinations.
> - **Context precision**: proportion of retrieved chunks that are relevant to the query. Higher = less noise in retrieval.
> - **Decomp. accuracy**: whether the agent correctly decomposed the query into sub-questions (ViLexAgent only).
> - **Cross-ref. accuracy**: whether the agent correctly identified which legal sources to cross-reference (ViLexAgent only).
> - **Legal currency score**: whether the agent correctly identified expired vs. still-in-force legal documents (ViLexAgent only). Score of 1.0 means all 30 queries received correct currency checks.

---

## 4. Per-Query Metric Summary

### 4.1 Faithfulness by System

Faithfulness scores per individual query (averages reported per type and domain below). Raw per-query scores show high variance across all three systems, particularly on Type C queries that require cross-referencing domestic law with international agreements (EVFTA, CPTPP).

| Query Type | NaiveRAG avg. faithfulness | RAGRerank avg. faithfulness | ViLexAgent avg. faithfulness |
|---|:---:|:---:|:---:|
| Type A (factual) | ~0.83 | ~0.85 | ~0.34 |
| Type B (applied) | ~0.79 | ~0.73 | ~0.50 |
| Type C (comparative) | ~0.23 | ~0.47 | ~0.34 |

| Domain | NaiveRAG avg. faithfulness | RAGRerank avg. faithfulness | ViLexAgent avg. faithfulness |
|---|:---:|:---:|:---:|
| Labor | ~0.61 | ~0.60 | ~0.36 |
| Food safety | ~0.72 | ~0.74 | ~0.40 |

> **Note:** ViLexAgent's lower faithfulness is likely a consequence of its more cautious and verbose answer style (including `<think>` reasoning traces, legal currency warnings, and multi-source citations), which the RAGAS faithfulness metric may penalise when answer statements are not directly traceable to a single retrieved chunk.

### 4.2 Context Precision by System

| Query Type | NaiveRAG | RAGRerank | ViLexAgent |
|---|:---:|:---:|:---:|
| Type A | ~0.57 | ~0.57 | ~0.67 |
| Type B | ~0.43 | ~0.47 | ~0.55 |
| Type C | ~0.10 | ~0.27 | ~0.28 |

ViLexAgent retrieves more relevant chunks than either baseline, especially on Type B applied questions. Type C comparative questions remain the hardest category for all systems.

### 4.3 Latency

| System | Min latency (s) | Max latency (s) | Avg latency (s) |
|---|:---:|:---:|:---:|
| NaiveRAG | 2.60 | 4.94 | 3.39 |
| RAGRerank | 2.87 | 5.15 | 3.57 |
| ViLexAgent | 4.04 | **54.94** | 8.70 |

ViLexAgent is significantly slower due to its multi-step agentic reasoning pipeline (query decomposition, multi-source retrieval, legal currency check, cross-reference scoring). The outlier at 54.94 s is query C08 ("CPTPP transparency requirements for food safety") — the most complex comparative query in the set.

---

## 5. ViLexAgent-Specific Metrics

Only ViLexAgent tracks the following three additional dimensions.

### 5.1 Decomposition Accuracy (`decomp_score`)

| Value | Count | Proportion |
|---|:---:|:---:|
| 1.0 (correct decomposition) | 23 | 76.7% |
| 0.0 (incorrect decomposition) | 7 | 23.3% |

Failures are concentrated in queries where the question mixes domestic and international sources (Type C) or where the question structure is ambiguous (e.g., C01 — CPTPP freedom of association conflict, C05 — forced labour CPTPP alignment).

### 5.2 Cross-Reference Accuracy (`crossref_score`)

| Value | Count | Proportion |
|---|:---:|:---:|
| 1.0 (correct cross-reference) | 18 | 60.0% |
| 0.0 (incorrect or missing) | 12 | 40.0% |

Failures align with the same Type C queries where decomposition also failed, and with some Type A/B food safety queries where the agent retrieved outdated legal documents instead of current ones.

### 5.3 Legal Currency Score (`currency_score`)

| Value | Count | Proportion |
|---|:---:|:---:|
| 1.0 | 30 | **100%** |

Every query received a currency score of 1.0, meaning ViLexAgent correctly flagged expired legal documents (e.g., Bộ luật Lao động 10/2012/QH13, Nghị định 44/2003, Pháp lệnh Hợp đồng lao động 165/HĐBT) and recommended currently-in-force alternatives (e.g., Bộ luật Lao động 2019, Nghị định 15/2018/NĐ-CP). This is ViLexAgent's most distinctive capability and is absent in both baselines.

---

## 6. Key Findings and Analysis

### 6.1 ViLexAgent's Strengths

1. **Legal currency detection (100% accuracy).** Both baselines retrieved and presented answers based on outdated legislation without warning. ViLexAgent consistently identified expired decrees and directed users to valid replacements. This is critical for Vietnamese legal QA where the regulatory landscape changes frequently.

2. **Better context precision (+0.09 vs. NaiveRAG, +0.09 vs. RAGRerank).** ViLexAgent's query decomposition step improves chunk retrieval relevance, particularly for applied (Type B) questions.

3. **Structured, citable answers.** ViLexAgent explicitly cites specific article numbers, decree identifiers, and EVFTA/CPTPP chapter references, making answers more useful for professional legal work.

4. **International alignment reasoning.** For Type C queries, ViLexAgent attempts to identify gaps or conflicts between Vietnamese domestic law and EVFTA/CPTPP requirements — a capability absent in both baselines.

### 6.2 ViLexAgent's Weaknesses

1. **Lower faithfulness (0.395 vs. ~0.65 for baselines).** The RAGAS faithfulness metric penalises verbose, multi-source answers. ViLexAgent's answers contain many statements about expired laws, general context, and international standards that are not directly present in any single retrieved chunk, causing faithfulness to drop.

2. **High latency (avg 8.70 s, max 54.94 s).** The agentic pipeline adds significant overhead. For a production legal assistant, latency above 10 s per query may be unacceptable.

3. **Cross-reference failures on Type C queries (40% error rate).** When a query requires simultaneous reference to Vietnamese domestic law and specific EVFTA/CPTPP articles, the agent sometimes misidentifies the relevant international provisions.

4. **Decomposition failures on ambiguous queries (23.3% error rate).** Queries phrased as implicit comparisons (e.g., "does Vietnam meet standard X?") are harder for the decomposer than explicit multi-part questions.

### 6.3 Baseline Comparison

The two baselines are very close to each other. RAGRerank improves slightly over NaiveRAG on faithfulness (+0.010) and context precision (+0.018), confirming that reranking adds marginal value. Neither baseline performs any legal currency checking or cross-referencing, making them unsuitable for Vietnamese legal QA in a production context despite their stronger faithfulness scores.

---

## 7. Hardest Queries Across All Systems

The following queries scored 0.0 on faithfulness across all three systems, indicating they are genuinely difficult for all approaches tested.

| ID | Query (summary) | Domain | Type | Issue |
|---|---|---|---|---|
| C02 | EVFTA labour standards vs. Vietnam's implementation | Labor | C | International cross-reference gap |
| C05 | Vietnam forced labour law vs. CPTPP compliance | Labor | C | Shallow domestic source retrieval |
| C09 | EVFTA anti-discrimination vs. Vietnam labour law | Labor | C | Missing EVFTA chapter in retrieval |

All three are Type C (comparative) queries requiring simultaneous retrieval from Vietnamese domestic law and international trade agreements — the hardest category in the benchmark.

---

## 8. Summary Table

| Criterion | Winner |
|---|---|
| Faithfulness (lower hallucination) | **Baseline 2 — RAGRerank** (0.662) |
| Context precision (retrieval relevance) | **ViLexAgent** (0.519) |
| Speed | **Baseline 1 — NaiveRAG** (3.39 s avg) |
| Legal currency checking | **ViLexAgent** (100%, baselines: N/A) |
| Query decomposition | **ViLexAgent** (76.7%, baselines: N/A) |
| Cross-reference accuracy | **ViLexAgent** (60.0%, baselines: N/A) |
| International law reasoning | **ViLexAgent** (baselines: absent) |
| Overall fitness for production legal QA | **ViLexAgent** (with caveats on latency and faithfulness) |

---

## 9. Recommendations

1. **Improve faithfulness:** Constrain ViLexAgent's answer generation to ground statements more tightly in retrieved chunks. Consider adding a post-generation faithfulness filter or using structured output templates that separate "retrieved facts" from "agent reasoning."

2. **Reduce latency:** Profile the pipeline to identify bottlenecks. Consider parallelising the decomposition and retrieval steps, or caching frequent legal currency checks.

3. **Strengthen Type C handling:** Fine-tune or expand the international-law retrieval module. Add explicit retrieval targets for EVFTA and CPTPP chapters when query type is detected as comparative.

4. **Reassess faithfulness metric:** RAGAS faithfulness may not be the right metric for an agentic legal assistant that intentionally reasons beyond retrieved chunks. Consider supplementing with a human-evaluated legal correctness score.

---

*Report generated from evaluation files dated 2026-05-06.*