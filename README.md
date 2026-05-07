# ⚖️ ViLexAgent

> **A Multi-Agent RAG System for Vietnamese Legal Q&A**  
> Supports Vietnamese labor law, food safety regulations, and international trade agreements (EVFTA, CPTPP).

---

## Overview

ViLexAgent is an agentic Retrieval-Augmented Generation (RAG) system that answers complex legal questions about Vietnamese law and its compliance with international trade agreements. The system decomposes user queries into sub-questions, retrieves relevant legal documents from a vector database, cross-references domestic law with international standards, and synthesizes a cited, legally-grounded answer.

**Key capabilities:**
- Query decomposition into domain-specific sub-questions (labor law, food safety)
- Hybrid retrieval from domestic Vietnamese legal documents and international treaty clauses (EVFTA, CPTPP)
- Automatic cross-reference analysis with alignment scoring (`aligned` / `conflict` / `gap` / `no_international`)
- Expired document detection with explicit warnings
- Step-by-step reasoning trace visible in the UI

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Decomposer  │  → Sub-questions + domain tags (labor / food_safety)
└─────────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌──────────────────┐        ┌──────────────────────────┐
│ Domestic         │        │ International             │
│ Retriever        │        │ Retriever                 │
│ (Qdrant + Jina)  │        │ (Qdrant + Jina)           │
└──────────────────┘        └──────────────────────────┘
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
        ┌─────────────────────┐
        │   Cross-Reference   │  → alignment: aligned / conflict / gap
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │     Synthesizer     │  → Final answer with citations
        └─────────────────────┘
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| UI | Chainlit 2.11 |
| Agent Orchestration | LangGraph 1.1 |
| Vector Database | Qdrant |
| Embedding Model | Jina Embeddings v5 Small (4-bit, CUDA) |
| LLM Backend | LiteLLM (OpenAI-compatible) |
| OCR / PDF Parsing | PaddleOCR, PyMuPDF |
| Vietnamese NLP | Underthesea |
| Experiment Tracking | MLflow |
| Evaluation | RAGAS |
| Runtime | Python 3.13, Poetry |

---

## Project Structure

```
vilexagent/
├── app/
│   └── vilexagent_ui.py       # Chainlit UI entry point
├── src/
│   ├── agents/
│   │   ├── state.py               # AgentState (LangGraph TypedDict)
│   │   ├── query_decomposer.py    # Query decomposition node
│   │   ├── domestic_retriever.py  # Vietnamese law retrieval node
│   │   ├── international_retriever.py  # EVFTA/CPTPP retrieval node
│   │   ├── cross_reference.py     # Cross-reference analysis node
│   │   ├── synthesizer.py         # Answer synthesis node
│   │   └── graph.py               # LangGraph pipeline definition
│   ├── ingestion/                 # Document ingestion pipeline
│   ├── retrieval/                 # Retrieval utilities
│   └── utils/
│       ├── llm.py                 # LLM client (LiteLLM wrapper)
│       ├── model_loader.py        # Singleton embedding model loader
│       ├── logger.py              # Loguru logger
│       └── json_utils.py          # Robust JSON extractor
├── evaluation/                    # RAGAS benchmark suite
├── docker/                        # Docker/Qdrant setup
├── tests/
├── pyproject.toml
└── chainlit.md
```

---

## Installation

### Prerequisites

- Python 3.13
- [Poetry](https://python-poetry.org/docs/#installation)
- CUDA-capable GPU (recommended: 4GB+ VRAM)
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally on port `6333`
- An OpenAI-compatible LLM API endpoint on port `3001`

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/dnAnh1523/vilexagent.git
cd vilexagent
```

**2. Install dependencies**
```bash
poetry install
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```env
LLM_BASE_URL=http://localhost:3001/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_model_name
```

**4. Start Qdrant**
```bash
docker compose -f docker/docker-compose.yml up -d
```

**5. Ingest documents**

Place your Vietnamese legal PDFs and international treaty documents in the `data/` directory, then run the ingestion pipeline (see `src/ingestion/`).

**6. Run the app**
```bash
poetry run chainlit run app/vilexagent_ui.py --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Usage

Once running, you can ask questions in Vietnamese about:

- **Labor law** — probationary periods, wrongful termination compensation, minimum wage, overtime
- **Food safety** — export conditions, hygiene standards, quarantine requirements
- **International compliance** — whether Vietnamese law meets EVFTA/CPTPP standards

The UI displays a step-by-step reasoning trace (collapsible) showing which documents were retrieved, cross-reference results, and alignment scores before the final answer.

**Example questions:**
```
Thời gian thử việc tối đa theo pháp luật lao động Việt Nam là bao lâu?

Nếu người sử dụng lao động đơn phương chấm dứt hợp đồng trái pháp luật
thì phải bồi thường những gì cho người lao động?

Việt Nam có đáp ứng các tiêu chuẩn lao động của CPTPP về tự do hiệp hội không?
```

---

## Configuration

Key settings in `chainlit.md` (UI welcome screen) and `.chainlit/config.toml`:

| Setting | Location | Description |
|---|---|---|
| `cot` | `config.toml [UI]` | Chain-of-thought display: `"full"` shows all steps |
| `session_timeout` | `config.toml [project]` | Session retention in seconds |
| `LLM_BASE_URL` | `.env` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `.env` | Model name passed to LiteLLM |

---

## Evaluation

The system includes a RAGAS-based evaluation suite in `evaluation/` with a benchmark of labeled legal Q&A pairs across three difficulty types:

- **Type A** — Single-domain factual queries (domestic law only)
- **Type B** — Multi-aspect domestic queries requiring reasoning across clauses
- **Type C** — Cross-reference queries requiring domestic + international alignment

Run evaluation:
```bash
poetry run python evaluation/run_evaluation.py
```

Results are tracked with MLflow. Start the MLflow UI:
```bash
poetry run mlflow ui
```

---

## License

This project is for educational and portfolio purposes.

---

*Built with LangGraph · Qdrant · Chainlit · Jina Embeddings*
