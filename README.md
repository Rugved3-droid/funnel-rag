# FUNNEL: A Three-Stage Agentic RAG Architecture for Clinical Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the reference implementation of the **FUNNEL** (Focused Unified Narrative Network via Evidential Layering) pipeline, a specialized Retrieval-Augmented Generation (RAG) architecture designed for clinical documentation support.

**📄 Publication:** This codebase is provided as supplementary material for reproducibility in support of our MethodsX manuscript.

## Overview

FUNNEL addresses the challenge of retrieving and synthesizing information from dense, longitudinal clinical timelines through a three-stage process:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FUNNEL PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Patient Records] ──► STAGE 1 ──► STAGE 2 ──► STAGE 3 ──► [Output]     │
│                        50 docs     10 docs     Cited Answer              │
│                                                                          │
│  Stage 1: Multi-Modal Retrieval (Vector + BM25 + Temporal)              │
│  Stage 2: LLM-Based Reranking (Gemini Flash 1.5)                        │
│  Stage 3: Evidence-Grounded Synthesis (Gemini Pro 1.5)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Multi-modal retrieval**: Combines vector similarity, BM25 keyword search, and temporal weighting
- **Temporal intelligence**: Document-type-specific decay functions prioritize recent clinical data
- **Mandatory citations**: Every claim in the output must be grounded in source documents
- **Conflict detection**: Automatically flags contradictory information (cosine similarity > 0.85 with negation signals)
- **Cost-efficient**: ~$0.04 per query with sub-2.5s latency

## Repository Structure

```
repro/
├── __init__.py           # Package initialization
├── config.py             # All configurable parameters (weights, thresholds, models)
├── prompts.py            # Exact prompt templates for reranking and synthesis
├── stage1_retrieval.py   # Multi-modal retrieval with temporal weighting
├── stage2_reranking.py   # LLM-based relevance scoring
├── stage3_synthesis.py   # Cited synthesis and conflict detection
├── demo.py               # End-to-end demonstration script
├── requirements.txt      # Python dependencies
├── LICENSE.txt           # MIT License
└── .gitignore            # Git ignore patterns
```

## Setup Instructions

### 1. Requirements
- Python 3.9+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/funnel-clinical-rag.git
cd funnel-clinical-rag

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Set your API key as an environment variable:

```bash
# Linux/Mac
export GOOGLE_API_KEY='your-api-key-here'

# Windows (PowerShell)
$env:GOOGLE_API_KEY='your-api-key-here'
```

## Usage

### Run the Demo

```bash
python -m repro.demo
```

### Use in Your Code

```python
from repro.stage1_retrieval import Stage1Retrieval
from repro.stage2_reranking import Stage2Reranking
from repro.stage3_synthesis import Stage3Synthesis

# Initialize pipeline
api_key = "your-api-key"
s1 = Stage1Retrieval(api_key)
s2 = Stage2Reranking(api_key)
s3 = Stage3Synthesis(api_key)

# Your clinical documents (list of dicts)
documents = [...]  # See format below

# Execute pipeline
query = "What is the patient's baseline creatinine?"
candidates = s1.retrieve(query, documents, top_k=50)
reranked, cost1 = s2.rerank(query, candidates, top_k=10)
conflicts = s3.detect_conflicts(reranked)
answer, citations, cost2 = s3.synthesize(query, reranked)
```

## Input Document Format

The pipeline expects clinical events as dictionaries:

```python
{
    "id": "event_uuid",                    # Unique identifier
    "type": "SOAP Notes",                  # Document type (see Table 1 in manuscript)
    "timestamp": datetime(2024, 1, 15),    # Document date
    "text": "Full document content...",    # Clinical text
    "embedding": [0.1, 0.2, ...]           # 768-dim vector (Gemini embedding-001)
}
```

### Supported Document Types

| Document Type | Priority Multiplier |
|---------------|---------------------|
| Discharge Summary | 1.5 |
| Procedure Notes | 1.3 |
| Imaging Reports | 1.2 |
| Progress Notes | 1.0 |
| Nursing Notes | 0.8 |

## Configuration Parameters

Key parameters can be adjusted in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEIGHT_SEMANTIC` | 0.5 | Vector similarity weight |
| `WEIGHT_KEYWORD` | 0.3 | BM25 keyword weight |
| `WEIGHT_PRIORITY` | 0.2 | Document priority weight |
| `TEMPORAL_LAMBDA` | 0.002 | Decay constant (~50% at 1 year) |
| `CONFLICT_SIMILARITY_THRESHOLD` | 0.85 | Cosine threshold for conflict detection |
| `TOP_K_STAGE1` | 50 | Candidates from Stage 1 |
| `TOP_K_RERANK` | 10 | Documents passed to Stage 3 |

## Performance

Based on preliminary validation (see manuscript):

| Metric | Value |
|--------|-------|
| Query Latency (P95) | < 2.5s |
| Cost per Query | ~$0.04 |
| Citation Accuracy | 100% (in test queries) |

## License

MIT License. See [LICENSE.txt](LICENSE.txt) for details.

## Citation

If you use this code in your research, please cite our MethodsX manuscript:

```bibtex
@article{parmar2025funnel,
  title={FUNNEL: A Three-Stage Agentic RAG Architecture for Clinical Admission Assessment with Mandatory Citation and Conflict Detection},
  author={Parmar, Rugved},
  journal={MethodsX},
  year={2025}
}
```

## Acknowledgements

- Anthropic Claude for manuscript preparation assistance
- Synthea for synthetic patient data generation
- Antigravity AI for development collaboration
