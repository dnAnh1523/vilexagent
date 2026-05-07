# Evaluation Results Summary Report

## Overview
This report summarizes the performance evaluation of three RAG-based systems: **Baseline1_NaiveRAG**, **Baseline2_RAGRerank**, and **ViLexAgent**. The evaluation is based on a set of automated metrics covering latency, accuracy, schema compliance, and output quality (relevance, groundedness, completeness).

## 1. General Metrics

| System | Total Questions | Error Rate | Avg Latency (s) |
|---|---|---|---|
| Baseline1_NaiveRAG | 30 | 0.0 | 3.53 |
| Baseline2_RAGRerank | 30 | 0.0 | 3.85 |
| ViLexAgent | 30 | 0.0 | 5.43 |

## 2. Tier 1 & Tier 2 Metrics (Schema & Routing)

*(Note: Specific architectural metrics are primarily evaluated for the agentic system)*

| System | T1 Pass Rate | T1 Citation Grounded | T1 Expired Warning | T2 Source Routing Acc | T2 Alignment Acc | T2 Keyword Coverage |
|---|---|---|---|---|---|---|
| Baseline1_NaiveRAG | N/A | N/A | N/A | N/A | N/A | 0.59 |
| Baseline2_RAGRerank | N/A | N/A | N/A | N/A | N/A | 0.59 |
| ViLexAgent | 1.00 | 0.80 | 1.00 | 0.80 | 0.70 | 0.64 |

## 3. Tier 3 Quality Metrics (1-5 Scale)

These metrics evaluate the final generated responses for relevance to the query, groundedness in the retrieved context, and completeness of the answer.

| System | Relevance (Avg) | Groundedness (Avg) | Completeness (Avg) |
|---|---|---|---|
| Baseline1_NaiveRAG | 4.43 | 2.73 | 3.93 |
| Baseline2_RAGRerank | 4.43 | 3.27 | 4.07 |
| ViLexAgent | 4.43 | 4.47 | 3.83 |

## 4. Key Takeaways
* **Groundedness Improvements**: The `ViLexAgent` demonstrates significantly higher Groundedness compared to the baselines. While `NaiveRAG` struggled with Groundedness (2.73) and `RAGRerank` improved it slightly (3.27), `ViLexAgent` achieved the highest score (4.47).
* **Latency Trade-off**: `ViLexAgent` has a higher average latency (~5.43s) compared to the `NaiveRAG` (~3.53s) and `RAGRerank` (~3.85s) baselines. This increased processing time is a direct trade-off for the advanced agentic processing, query decomposition, and routing tasks.
* **Agentic Capabilities**: `ViLexAgent` successfully executed complex architectural instructions with a 100% pass rate for both decomposition schema and expired document warnings. It also demonstrated strong source routing capabilities with an 80% accuracy rate.