<div align="center">

# 🌿 زرعة · Zar3a

### *Intelligent Urban Forestry & Agricultural AI Platform for Egypt*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.13%2B-DC143C?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Grafana](https://img.shields.io/badge/Grafana-11.2-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.16%2B-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Zar3a (زرعة)** is a production-grade, end-to-end Agentic AI and RAG platform purpose-built for **smart urban agriculture and forestry in Egypt**. It empowers citizens, urban planners, and environmental agencies with AI-driven answers grounded in official Egyptian environmental law, real microclimate ML predictions, and intelligent tree recommendation science.

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [High-Level Architecture](#-high-level-architecture)
3. [End-to-End Data Ingestion Pipeline](#-end-to-end-data-ingestion-pipeline)
4. [Advanced Chunking & Vector Database](#-advanced-chunking--vector-database-qdrant)
5. [Agentic Workflow & RAG](#-agentic-workflow--retrieval-augmented-generation)
6. [ML Features & Recommendation System](#-ml-features--recommendation-system)
7. [Full-Stack Observability](#-full-stack-observability)
8. [Project Structure](#-project-structure)
9. [Technology Stack](#-technology-stack)
10. [Getting Started](#-getting-started)
11. [Environment Variables](#-environment-variables)
12. [API Reference](#-api-reference)
13. [Testing](#-testing)
14. [Contributing](#-contributing)

---

## 🌍 Project Overview

**Zar3a** (Arabic: زرعة, meaning *"planting"*) is an intelligent decision-support system that bridges official Egyptian environmental legislation, urban heat island science, and AI-powered knowledge retrieval. The platform answers complex, domain-specific questions such as:

> *"What tree species can I legally plant on a narrow Cairo sidewalk that also mitigates the urban heat island effect?"*

To answer this, Zar3a concurrently:
- Retrieves relevant passages from **official Egyptian environmental laws** (e.g., Environment Law No. 4/1994, Ministerial Decree 147) stored in its semantic vector database
- Runs a **microclimate ML model** to predict temperature delta and NDVI improvement scores
- Applies the **Tree Recommendation Engine** to filter and rank candidate species by site constraints
- **Synthesizes** all evidence into a single, grounded, Arabic-first bilingual response

### Core Capabilities

| Capability | Description |
|---|---|
| 🏛️ **Legal Knowledge RAG** | Retrieves grounded answers from Egyptian forestry laws, decrees & environmental regulations |
| 🌡️ **Microclimate ML** | Predicts urban heat island temperature deltas and NDVI impact per candidate site |
| 🌳 **Tree Recommendation** | Filters and ranks tree species by street width, water requirement & cooling efficiency |
| 🔍 **Hybrid Search** | Dense E5 embeddings + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF) |
| 🤖 **Agentic Orchestration** | LangGraph multi-agent graph with parallel fan-out and MemorySaver multi-turn persistence |
| 📊 **Observability Stack** | Real-time metrics (Prometheus), structured log aggregation (Loki + Promtail), dashboards (Grafana) |
| 🧠 **Dual LLM Mode** | Seamlessly falls back: OpenAI (GPT-4o-mini) → Local Ollama (Qwen2.5:3b) → Keyword routing |
| 🌐 **Arabic-First** | Full Arabic NLP: text normalization, bidirectional rendering, Arabic keyword routing |

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER / FRONTEND (Nginx)                              │
│                       http://localhost:8888                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTP/REST
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend  :8000                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   LangGraph Multi-Agent Graph                       │   │
│   │                                                                     │   │
│   │  START → [Supervisor Router] ──────────────────────────────────►   │   │
│   │                   │                                                │   │
│   │         ┌─────────┴──────────┐                                    │   │
│   │         ▼                    ▼                                    │   │
│   │  [Knowledge Node]   [Climate Rec Node]   ← parallel fan-out       │   │
│   │  RAG Retriever        ML Predictor +                              │   │
│   │  Qdrant dense+BM25    Tree Recommender                            │   │
│   │         │                    │                                    │   │
│   │         └─────────┬──────────┘                                    │   │
│   │                   ▼                                                │   │
│   │           [Synthesizer Node]  ← merges all evidence               │   │
│   │                   │                                                │   │
│   │                  END → final_response                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   /metrics ──► Prometheus :9090                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                      │                     │
          ▼                      ▼                     ▼
   ┌─────────────┐    ┌──────────────────┐   ┌──────────────┐
   │   Qdrant    │    │   PostgreSQL     │   │    Loki      │
   │ :6333/6334  │    │  (Chunk Store)   │   │   :3100      │
   │ Vector DB   │    │  Relational DB   │   │ Log Aggreg.  │
   └─────────────┘    └──────────────────┘   └──────────────┘
                                                      ▲
                                              ┌───────┴──────┐
                                              │   Promtail   │
                                              │ (Log Shipper)│
                                              └──────────────┘

   ┌─────────────────────────────────────────────────────┐
   │              Grafana Observability  :3000            │
   │   Dashboards ← Prometheus metrics + Loki logs        │
   └─────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Raw Documents / URLs
       │
       ▼
[Ingestion Pipeline]
│  ├─ DirectWebScraper    (WikiSource, legal HTML pages)
│  ├─ PDFDownloaderScraper (Manshurat, paginated PDFs)
│  └─ LocalPDFLoader      (on-disk Arabic PDF files)
       │
       ▼
[OCR Engine]
│  ├─ LocalPDFTextExtractor  (pypdf — zero API cost for digital PDFs)
│  ├─ PaddleOCRExtractor     (offline, Arabic lang=ar, PyMuPDF rendering)
│  └─ MistralOCRClient       (cloud API fallback, key pooling + backoff)
       │
       ▼
[Text Normalization]
│  └─ ArabicTextNormalizer (Hamza, Tashkeel, diacritics, NFKC)
       │
       ▼
[Chunking Pipeline]
│  ├─ SemanticChunker     (article-header split → recursive fallback)
│  ├─ OverlappingChunker  (sliding window with configurable overlap)
│  └─ ChunkIDGenerator    (SHA-256 deterministic deduplication)
       │
       ▼
[PostgreSQL] ──── [RAG Ingestion Pipeline] ──── [Qdrant]
 chunks table      RAGEmbedder                   zar3a_knowledge_base
 embedding_status  (E5-large multilingual)       dense vectors 1024d
 vector_id         batch upsert (size=50)        + sparse BM25 index
```

---

## 📥 End-to-End Data Ingestion Pipeline

The ingestion pipeline is a fully async, multi-stage orchestrator residing in `back_end/src/ingestion/`.

### Stage 1 — Web Scrapers

| Scraper | File | Source Type | Output |
|---|---|---|---|
| **DirectWebScraper** | `web_scraper_direct.py` | HTML pages (WikiSource, official portals) | Structured JSON with Arabic legal text |
| **PDFDownloaderScraper** | `web_scraper_pdf.py` | Paginated PDF catalogue pages (Manshurat) | Raw PDFs → `Data/Arabic_Data/` |
| **LocalPDFLoader** | `local_pdf_loader.py` | On-disk PDF file paths | Passes files to OCR engine |

### Stage 2 — Hybrid OCR Engine

The `MistralPDFProcessor` is a resilient facade that selects the optimal OCR strategy per document:

```python
# Strategy selection (automatic):
# 1. LocalPDFTextExtractor  — if digital text > min_char_threshold (100 chars/page)
# 2. PaddleOCRExtractor     — offline, Arabic-trained PaddleOCR with PyMuPDF rendering
# 3. MistralOCRClient       — cloud API fallback with exponential backoff + key rotation
```

#### OCR Components

| Component | Technology | Description |
|---|---|---|
| **LocalPDFTextExtractor** | `pypdf` | Extracts selectable text — zero API cost |
| **PaddleOCRExtractor** | `PaddleOCR` (lang=`ar`) + `PyMuPDF` | Offline Arabic OCR; env flags: `FLAGS_use_mkldnn=0`, `FLAGS_enable_pir_api=0` |
| **APIKeyPoolManager** | Custom | Rotates Mistral API keys with per-key cooldown and health tracking |
| **MistralOCRClient** | `mistralai` SDK | Cloud OCR with exponential backoff, jitter, `Retry-After` compliance |

### Stage 3 — Arabic Text Normalization

The `ArabicTextNormalizer` applies a comprehensive normalization pipeline:

- Hamza normalization (أ، إ، آ → ا)
- Tashkeel (diacritics) removal
- Tatweel/Kashida removal
- Unicode NFKC normalization
- Sentence boundary and whitespace cleanup
- Optional Ollama-powered Arabic grammar correction (`OllamaArabicCorrector`)

---

## 🧩 Advanced Chunking & Vector Database (Qdrant)

### Chunking Strategies

All chunkers implement the `BaseChunker` interface and are orchestrated by the `ChunkingPipeline`.

#### SemanticChunker — Article-Aware Two-Pass Splitting

The primary chunker for Arabic legal documents uses a two-pass strategy:

**Pass 1 — Structural Split (Article Headers)**
```
Splits on: المادة / مادة / Article (N)
→ Each article becomes its own chunk with metadata: type, header, article_number
→ Ideal for legal citations: "According to Article 7 of Law 4/1994..."
```

**Pass 2 — Recursive Character Fallback**
```
Triggered when Pass 1 yields ≤ 1 chunk (no structural headers found)
Splitter hierarchy: paragraph → sentence → word boundary
Covers: OCR output, ministry announcements, wiki scrapes
```

Each chunk carries strongly-typed `ChunkMetadata`:

```python
class ChunkMetadata(BaseModel):
    type: str           # 'preamble' | 'article' | 'section' | 'paragraph' | 'recursive'
    header: Optional[str]
    article_number: Optional[Union[int, str]]
    word_count: int
    language: str       # 'ar' | 'en'
```

#### OverlappingChunker

Sliding window chunker with configurable overlap for non-structured documents, ensuring dense context coverage for retrieval-focused use cases.

#### ChunkIDGenerator — Deterministic Deduplication

Generates SHA-256-based deterministic chunk IDs from content + metadata, enabling idempotent re-ingestion and deduplication at the database level.

### Vector Database — Qdrant

| Setting | Value |
|---|---|
| **Collection** | `zar3a_knowledge_base` |
| **Embedding Model** | `intfloat/multilingual-e5-large` |
| **Dimensions** | 1024 |
| **Batch Size** | 50 chunks |
| **Top-K Retrieval** | 5 results |
| **RRF Constant** | 60 |
| **Candidate Pool** | Top-K × 4 (pre-filter) |

The `VectorStore` uses a **process-wide singleton** pattern (`threading.Lock`) to prevent the Qdrant local storage exclusive-lock conflict when multiple code paths access it concurrently.

#### RAG Ingestion Pipeline

```
PostgreSQL (un-embedded chunks)
         │
         ▼  get_chunks_with_metadata_for_vectorization()
[RAGEmbedder.embed_texts()]  ← is_query=False (passage mode)
         │
         ▼  E5-large multilingual embeddings (batch of 50)
[VectorStore.upsert_chunks()]  → Qdrant upsert
         │
         ▼  update_chunk_embedding_status() + update_chunk_vector_id()
PostgreSQL (is_embedded=True, vector_id stored)
```

### Hybrid Retrieval (Dense + Sparse + RRF)

The `RAGRetriever` implements a three-stage hybrid search:

```
query → [E5 embed] ─────────────────► Qdrant dense search (top K×4)
                                               │
query → [BM25 index] ────────────────► BM25 keyword search (top K×4)
                                               │
                                     [RRF Fusion (k=60)]
                                               │
                                       Final top-K chunks
```

---

## 🤖 Agentic Workflow & Retrieval-Augmented Generation

The entire multi-agent workflow is implemented as a **LangGraph `StateGraph`** — a directed, stateful computation graph with parallel fan-out support.

### Agent State

All nodes share a single `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], add_messages]  # conversation history
    routes:            List[str]                # router decision(s)
    rag_context:       Optional[str]            # retrieved legal text passages
    climate_metrics:   Optional[Dict[str, Any]] # ML model outputs
    candidate_species: Optional[List[Dict]]     # ranked tree recommendations
    language:          str                      # 'ar' | 'en'
    final_response:    Optional[str]            # final synthesized answer
```

### LangGraph Workflow

```
              START
                │
                ▼
       ┌─────────────────┐
       │  Supervisor      │  LLM: GPT-4o-mini or Qwen2.5:3b
       │  Router Node     │  temp=0.0, RouterOutput structured schema
       └────────┬────────┘  Fallback: Arabic+English keyword matching
                │
       ┌────────┴────────────────────────┐
       │                                 │
       ▼  "knowledge"                    ▼  "climate_recommendation"
┌──────────────────┐           ┌─────────────────────────────┐
│  Knowledge Node  │           │  Climate Recommendation Node │
│                  │           │                              │
│  1. RAGRetriever │           │  1. Climate ML Predictor     │
│     (E5 + BM25)  │           │     (XGBoost/LightGBM/RF)   │
│  2. Format       │           │  2. TreeRecommendationEngine │
│     rag_context  │           │     (criteria filtering +    │
│  3. LLM cite     │           │      weighted scoring)       │
│     synthesis    │           │  3. LLM recommendation       │
└──────────┬───────┘           └──────────────┬───────────────┘
           │                                  │
           └────────────────┬─────────────────┘
                            ▼
                  ┌──────────────────┐
                  │  Synthesizer     │  Merges legal citations +
                  │  Node            │  climate metrics +
                  └────────┬─────────┘  species recommendations
                           │
                          END → final_response (Arabic / English)
```

### Supervisor Router — Dual-Mode Intelligence

The router operates in two modes:

| Mode | Trigger | Description |
|---|---|---|
| **LLM Mode** | `OPENAI_API_KEY` set | GPT-4o-mini with structured `RouterOutput` schema at `temperature=0.0` |
| **Keyword Mode** | No API key / Ollama unavailable | Bilingual keyword matching with Arabic morphological prefix stripping |

**Routing Destinations:**

| Route | Triggers On |
|---|---|
| `"knowledge"` | Laws, decrees, regulations, environmental policy, COP agreements, climate strategy |
| `"climate_recommendation"` | Tree species, shade, UHI, planting, drought tolerance, narrow streets |
| **Both (parallel)** | Compound queries, ambiguous intent — always the safe default |

### LLM Factory — Graceful Degradation

```python
# Priority order for LLM resolution (build_chat_llm):
# 1. Real OPENAI_API_KEY → OpenAI ChatOpenAI (GPT-4o-mini)
# 2. Local Ollama reachable → ChatOpenAI via Ollama /v1 compat (Qwen2.5:3b)
# 3. Neither → Returns None → keyword routing only (zero-dependency mode)
```

The Ollama availability check uses a **60-second TTL cache** to avoid blocking API paths.

### Multi-Turn Memory

The graph is compiled with `MemorySaver` for in-process persistence, or any external `BaseCheckpointSaver` for production:

```python
# In-memory (development):
compiled_graph = create_agent_graph(checkpointer=MemorySaver())

# Persistent (production):
compiled_graph = create_agent_graph(checkpointer=PostgresSaver(...))
```

---

## 📊 ML Features & Recommendation System

### Climate ML Pipeline

> **Directory:** `back_end/src/climate_ml/`

A fully reproducible, MLflow-tracked training system managed with **DVC** and **Hydra**.

#### Dataset

- **Source:** `cairo_comprehensive_master_dataset.csv`
- **Target:** Temperature delta (°C) from urban tree canopy interventions
- **Features:** Geospatial attributes, NDVI, street morphology, soil type, canopy density

#### Model Benchmarking

Three model classes are benchmarked per training run:

| Model | Library | MLflow Integration |
|---|---|---|
| **XGBoost** | `xgboost>=3.2` | `mlflow.xgboost` autolog |
| **LightGBM** | `lightgbm>=4.7` | `mlflow.lightgbm` autolog |
| **Random Forest** | `sklearn` | `mlflow.sklearn` autolog |

Training is orchestrated via **Hydra** (`conf/config.yaml`) and tracked to **DagsHub**:

```bash
# Run the DVC-managed training pipeline:
cd back_end/src/climate_ml
dvc repro
```

**MLflow tracking** captures: hyperparameters, RMSE/MAE/R², feature importances, model artifacts, and input signatures.

#### Runtime Climate Predictions

```python
# AgentState.climate_metrics output structure:
{
    "predicted_temp_delta_c": -2.3,   # cooling effect in °C
    "ndvi_improvement": 0.12,         # greenness index change
    "cooling_confidence": 0.87,       # model confidence score
}
```

### Tree Recommendation Engine

> **File:** `recommendation_system/engine.py`

The `TreeRecommendationEngine` is a **Singleton** backed by a curated `trees_data.csv` dataset of Egyptian urban tree species.

#### Multi-Criteria Filtering

```python
criteria = {
    "narrow_street":     True,    # suitable_for_narrow_streets == True
    "water_requirement": "Low",   # 'Low' | 'Medium' | 'High' (inclusive)
    "min_cooling_score": 7.0,     # cooling_effect_score >= threshold
    "top_n":             5,
    "cooling_weight":    0.6,
    "carbon_weight":     0.4,
}
```

#### Weighted Scoring Formula

```
final_score = (cooling_effect_score × cooling_weight)
            + (carbon_sequestration_kg_yr × carbon_weight)
```

**Example top recommendation:**
```
🌳 نيم (Neem) | Shade Category
   Water: Medium | Cooling Score: 8 | Final Score: 19.08
```

---

## 📡 Full-Stack Observability

The platform ships a complete observability stack — no additional configuration required.

### Stack Components

| Component | Image | Port | Role |
|---|---|---|---|
| **Prometheus** | `prom/prometheus:latest` | `9090` | Metrics scraping & time-series storage |
| **Loki** | `grafana/loki:latest` | `3100` | Structured log aggregation |
| **Promtail** | `grafana/promtail:latest` | — | Container log shipping → Loki |
| **Grafana** | `grafana/grafana:11.2.0` | `3000` | Unified metrics + logs dashboards |

### Prometheus Scrape Targets

```yaml
scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "zar3a_backend"
    static_configs:
      - targets: ["backend:8000"]
    metrics_path: /metrics       # prometheus-fastapi-instrumentator
```

The `/metrics` endpoint exposes:
- HTTP request count, latency histograms, and error rates per endpoint
- Active connections and response size distributions

### Grafana Quick Start

1. Navigate to **[http://localhost:3000](http://localhost:3000)** → `admin` / `admin`
2. Add **Prometheus** data source: `http://prometheus:9090`
3. Add **Loki** data source: `http://loki:3100`
4. Import recommended dashboards:
   - **Loki Logs Explorer** — filter by `container=zar3a_backend`, search by log level
   - **FastAPI Metrics** — P95/P99 latencies, throughput, error rates
   - **Prometheus Self-Monitoring** — scrape health and target status

---

## 📁 Project Structure

```
Zar3a/
├── docker-compose.yml              # Full-stack orchestration (8 services)
├── pyproject.toml                  # Python dependencies (uv)
├── .env                            # Environment variables (gitignored)
├── .env.example                    # Environment template
│
├── back_end/
│   ├── Dockerfile                  # python:3.11-slim + uv single-stage build
│   └── src/
│       ├── api/                    # FastAPI application & routers
│       ├── agent/
│       │   ├── graph.py            # LangGraph StateGraph definition
│       │   ├── router.py           # Supervisor router (LLM + keyword fallback)
│       │   ├── state.py            # AgentState TypedDict
│       │   ├── config.py           # LLM factory (OpenAI / Ollama / None)
│       │   └── nodes/
│       │       ├── knowledge_node.py       # RAG retrieval + legal synthesis
│       │       ├── climate_rec_node.py     # ML prediction + tree recommendation
│       │       └── synthesizer_node.py     # Final response synthesis
│       ├── ingestion/
│       │   ├── ingestion_pipeline.py       # Async pipeline orchestrator
│       │   ├── web_scraper_direct.py       # HTML / WikiSource scraper
│       │   ├── web_scraper_pdf.py          # Paginated PDF downloader
│       │   ├── local_pdf_loader.py         # On-disk PDF ingestion
│       │   ├── ocr_loader.py               # Hybrid OCR engine
│       │   ├── arabic_text_normalizer.py   # Arabic NLP normalization
│       │   └── ollama_arabic_corrector.py  # LLM grammar correction
│       ├── chunkers/
│       │   ├── ChunkingPipeline.py         # Chunker orchestrator
│       │   ├── semantic_chunker.py         # Article-aware + RCT fallback
│       │   ├── overlapping_chunker.py      # Sliding window chunker
│       │   └── chunk_id_generator.py       # SHA-256 deduplication
│       ├── rag_database/
│       │   ├── rag_pipeline.py     # PostgreSQL → Embedder → Qdrant
│       │   ├── vector_store.py     # Qdrant singleton client
│       │   ├── retriever.py        # Hybrid retriever (E5 + BM25 + RRF)
│       │   ├── embedder.py         # E5-large multilingual embedder
│       │   └── config.py           # RAGSettings (pydantic-settings)
│       ├── recommendation_system/
│       │   ├── engine.py           # TreeRecommendationEngine (Singleton)
│       │   ├── tool.py             # LangGraph tool wrapper
│       │   └── data/trees_data.csv # Egyptian urban tree dataset
│       ├── climate_ml/
│       │   ├── dvc.yaml            # DVC pipeline definition
│       │   ├── train_pipeline.py   # Hydra + MLflow training
│       │   ├── conf/config.yaml    # Hydra configuration
│       │   └── src/
│       │       ├── trainer.py      # XGBoost / LightGBM / RF trainer
│       │       ├── predictor.py    # Runtime inference wrapper
│       │       ├── evaluation.py   # RMSE / MAE / R² evaluation
│       │       └── data_loader.py  # Cairo climate dataset loader
│       └── database/
│           └── operations.py       # SQLAlchemy CRUD operations
│
├── frontend/
│   ├── Dockerfile                  # Nginx static file server
│   ├── index.html                  # Single-page chat interface
│   ├── css/ / js/ / assets/
│
├── observability/
│   ├── prometheus.yml              # Prometheus scrape configuration
│   └── promtail-config.yml         # Docker log shipping config
│
└── tests/
    ├── test_api_chat.py
    ├── test_agent_config.py
    └── test_e2e_rag_flow.py
```

---

## 🛠️ Technology Stack

### Backend & AI

| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | 0.100+ | Async REST API server |
| **LangGraph** | 0.2+ | Multi-agent stateful graph |
| **LangChain OpenAI** | 0.2+ | LLM integrations (OpenAI + Ollama) |
| **Qdrant Client** | 1.13+ | Vector database client |
| **Sentence Transformers** | 3.0+ | E5-large multilingual embeddings |
| **XGBoost / LightGBM** | 3.2+ / 4.7+ | Microclimate ML models |
| **PaddleOCR** | 2.9.1 | Arabic offline OCR |
| **PyMuPDF** | 1.28+ | PDF rendering for OCR |
| **mistralai** | 2.10+ | Cloud OCR API |
| **SQLAlchemy** | 2.0+ | PostgreSQL ORM |
| **MLflow** | 3.16+ | ML experiment tracking |
| **DVC** | 3.67+ | ML pipeline versioning |
| **Hydra** | 1.3+ | ML configuration management |
| **pydantic-settings** | 2.0+ | Type-safe environment config |

### Infrastructure

| Technology | Version | Purpose |
|---|---|---|
| **Docker Compose** | V2 | Multi-container orchestration |
| **Nginx** | Latest | Frontend static serving |
| **Prometheus** | Latest | Metrics time-series |
| **Grafana** | **11.2.0** | Observability dashboards |
| **Loki** | Latest | Log aggregation |
| **Promtail** | Latest | Log shipping |
| **uv** | Latest | Ultra-fast Python package manager |

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL2 backend on Windows)
- [Git](https://git-scm.com/)
- *(Optional)* OpenAI API key — the system works without one via Ollama or keyword routing

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Zar3a.git
cd Zar3a
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Start the Full Stack

```bash
docker compose up -d
```

This starts **8 services** in dependency order:

```
qdrant → prometheus → loki → promtail → backend → grafana → frontend
```

### 4. Verify All Services Are Running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 5. Access the Platform

| Service | URL | Notes |
|---|---|---|
| 🌐 **Frontend Chat UI** | http://localhost:8888 | Main user interface |
| ⚡ **Backend API** | http://localhost:8000 | FastAPI JSON REST |
| 📖 **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API explorer |
| 📊 **Grafana Dashboards** | http://localhost:3000 | `admin` / `admin` |
| 🔥 **Prometheus** | http://localhost:9090 | Metrics explorer |
| 🗄️ **Qdrant UI** | http://localhost:6333/dashboard | Vector DB inspector |

### 6. Populate the Knowledge Base (Optional)

```bash
# Enter the backend container
docker exec -it zar3a_backend bash

# Run the full ingestion pipeline
python src/ingestion/ingestion_pipeline.py

# Index all chunks into Qdrant
python -c "
from src.rag_database.rag_pipeline import RAGIngestionPipeline
from src.database.session import get_db
pipeline = RAGIngestionPipeline()
with next(get_db()) as db:
    count = pipeline.process_unembedded_chunks(db)
    print(f'Indexed {count} chunks into Qdrant')
"
```

### Useful Commands

```bash
# View live backend logs
docker logs -f zar3a_backend

# Stop all services (preserve data volumes)
docker compose down

# Full reset (destroys all volumes)
docker compose down -v

# Rebuild backend after code changes
docker compose up -d --build backend

# Force recreate a specific service
docker compose up -d --force-recreate grafana
```

---

## 🔧 Environment Variables

```dotenv
# ── LLM Configuration ──────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...                    # Leave empty to use local Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
AGENT_LLM_MODEL=qwen2.5:3b

# ── RAG / Vector Database ──────────────────────────────────────────────────────
VECTOR_DB_HOST=qdrant
VECTOR_DB_PORT=6333
VECTOR_DB_COLLECTION_NAME=zar3a_knowledge_base
# QDRANT_LOCAL_PATH=./qdrant_storage    # For local dev without Docker

# ── Embedding Model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
EMBEDDING_DIMENSION=1024

# ── Retrieval Tuning ───────────────────────────────────────────────────────────
RETRIEVER_TOP_K=5
RRF_K=60
RAG_BATCH_SIZE=50

# ── OCR (optional) ─────────────────────────────────────────────────────────────
MISTRAL_API_KEY=...

# ── MLflow / DagsHub Tracking ─────────────────────────────────────────────────
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/Zar3a.mlflow
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-token
```

---

## 📖 API Reference

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
  "message": "ما هي الأشجار المناسبة للزراعة في الشوارع الضيقة بالقاهرة؟",
  "thread_id": "user-session-001",
  "language": "ar"
}
```

**Response:**
```json
{
  "response": "بناءً على المادة 7 من قانون البيئة رقم 4 لسنة 1994...",
  "routes_taken": ["knowledge", "climate_recommendation"],
  "candidate_species": [
    {
      "tree_id": 7,
      "name_ar": "نيم",
      "name_en": "Neem",
      "category": "Shade",
      "water_requirement": "Medium",
      "cooling_effect_score": 8,
      "final_score": 19.08
    }
  ],
  "climate_metrics": {
    "predicted_temp_delta_c": -2.3,
    "ndvi_improvement": 0.12
  }
}
```

### Health Check

```http
GET /health
→ {"status": "healthy", "version": "0.1.0"}
```

### Prometheus Metrics

```http
GET /metrics
→ Prometheus text format (HTTP counters, latency histograms)
```

---

## 🧪 Testing

```bash
# Install dependencies
uv pip install --system -r pyproject.toml

# Run full test suite
pytest tests/ -v

# Run specific test module
pytest tests/test_api_chat.py -v

# End-to-end RAG flow test
python test_e2e_rag_flow.py
```

**Test coverage includes:**
- `test_api_chat.py` — FastAPI endpoint integration tests
- `test_agent_config.py` — LLM factory and Ollama availability tests
- `test_e2e_rag_flow.py` — End-to-end ingestion → retrieval → generation flow

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a branch: `git checkout -b feature/my-new-feature`
3. **Test** your changes: `pytest tests/ -v`
4. **Commit**: `git commit -m "feat: add X capability"`
5. Open a **Pull Request**

### Code Style Guidelines

- Python 3.11+ type annotations throughout
- `pydantic` models for all structured data
- Structured logging: `logging.getLogger("Zar3a.<Module>")` — no bare `print()` calls
- Every module must have a docstring describing its role in the system

---

<div align="center">

Built with ❤️ for **smarter, greener Egyptian cities**

**زرعة** · *Plant a tree. Change a city.*

</div>
