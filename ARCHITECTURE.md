# EEGDash LLM Tagger - Architecture

This document describes the architecture of the EEGDash LLM Tagger system, which automatically classifies EEG/MEG neuroimaging datasets using Large Language Models.

## System Overview

The system consists of two repositories:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EEGDash Ingestion Pipeline                        │
│                                                                             │
│   fetch → clone → digest → [3.5_tag] → validate → inject                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         eegdash-llm-api (API Service)                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │  FastAPI    │───▶│   Orchestrator   │───▶│   Content-Addressable   │    │
│  │  Endpoints  │    │                  │    │        Cache            │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘    │
│                              │                                              │
└──────────────────────────────│──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     eegdash-llm-tagger (Core Library)                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │  OpenRouter │    │  BIDS Metadata   │    │    Abstract Fetcher     │    │
│  │   Tagger    │    │     Parser       │    │  (CrossRef/PubMed/S2)   │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘    │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bundled Config: prompt.md + few_shot_examples.json                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   OpenRouter.ai     │
                    │   (GPT-4/Claude)    │
                    └─────────────────────┘
```

## Repository Structure

### 1. eegdash-llm-tagger (Core Library)

A pip-installable Python package containing the core tagging logic.

```
eegdash-llm-tagger/
├── src/eegdash_tagger/
│   ├── tagging/
│   │   ├── llm_tagger.py      # OpenRouterTagger - LLM classification
│   │   └── tagger.py          # Protocol definitions
│   ├── metadata/
│   │   └── parser.py          # DatasetSummary - BIDS parsing
│   └── scraping/
│       ├── scraper.py         # GitHub cloning, OpenNeuro GraphQL
│       ├── enrichment.py      # Metadata enrichment orchestration
│       └── abstract_fetcher.py # DOI extraction, paper abstract fetching
├── prompt.md                   # System prompt for LLM
├── data/processed/
│   └── few_shot_examples.json  # Labeled examples for in-context learning
└── pyproject.toml              # Package configuration
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `OpenRouterTagger` | Sends metadata to LLM via OpenRouter API, parses structured response |
| `DatasetSummary` | Parses BIDS datasets (description, README, participants, tasks, events) |
| `fetch_abstract_with_cache` | Fetches paper abstracts from CrossRef/Semantic Scholar/PubMed |

### 2. eegdash-llm-api (API Service)

A FastAPI service that wraps the core library with caching and orchestration.

```
eegdash-llm-api/
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI endpoints
│   └── services/
│       ├── cache.py           # Content-addressable caching
│       └── orchestrator.py    # Clone → Parse → Cache → Tag workflow
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `TaggingCache` | Content-addressable cache with automatic invalidation |
| `TaggingOrchestrator` | Orchestrates the full tagging workflow |

## Data Flow

### Request Flow

```
1. POST /api/v1/tag
   {
     "dataset_id": "ds001234",
     "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
   }

2. Orchestrator:
   ├── Clone repository (shallow, to temp directory)
   ├── Extract BIDS metadata using DatasetSummary
   ├── Fetch paper abstract (if DOI found in references)
   ├── Build cache key: {dataset_id}:{metadata_hash}:{config_hash}:{model}
   ├── Check cache:
   │   ├── HIT → Return cached result
   │   └── MISS → Continue
   ├── Call OpenRouterTagger.tag_with_details()
   ├── Store result in cache
   └── Return result

3. Response:
   {
     "dataset_id": "ds001234",
     "pathology": ["Healthy"],
     "modality": ["Visual"],
     "type": ["Perception"],
     "confidence": {"pathology": 0.9, "modality": 0.85, "type": 0.8},
     "from_cache": false
   }
```

### Cache Key Structure

```
{dataset_id}:{metadata_hash}:{config_hash}:{model}
     │              │              │           │
     │              │              │           └── LLM model (e.g., openai/gpt-4-turbo)
     │              │              │
     │              │              └── SHA-256 of prompt.md + few_shot_examples.json
     │              │                  (changes when prompts are updated)
     │              │
     │              └── SHA-256 of filtered metadata fields
     │                  (changes when dataset content changes)
     │
     └── Dataset identifier (e.g., ds001234)
```

**Automatic Invalidation:**
- Update `prompt.md` → `config_hash` changes → new LLM calls
- Update `few_shot_examples.json` → `config_hash` changes → new LLM calls
- Dataset content changes → `metadata_hash` changes → new LLM call

## LLM Classification

### Input to LLM

```json
{
  "few_shot_examples": [
    {
      "pathology": ["Healthy"],
      "modality": ["Visual"],
      "type": ["Perception"],
      "metadata": {
        "title": "Visual perception study",
        "readme": "Participants viewed images..."
      }
    }
  ],
  "dataset": {
    "title": "New dataset to classify",
    "dataset_description": "...",
    "readme": "...",
    "participants_overview": "...",
    "tasks": [...],
    "events": [...],
    "paper_abstract": "..."
  }
}
```

### Output from LLM

```json
{
  "pathology": ["Healthy"],
  "modality": ["Visual", "Auditory"],
  "type": ["Perception"],
  "confidence": {
    "pathology": 0.95,
    "modality": 0.80,
    "type": 0.85
  },
  "reasoning": {
    "few_shot_analysis": "Similar to example X...",
    "metadata_analysis": "README mentions 'visual stimuli'...",
    "paper_abstract_analysis": "Abstract confirms perception study...",
    "decision_summary": "Classified as visual perception based on..."
  }
}
```

### Classification Categories

| Category | Values |
|----------|--------|
| **Pathology** | Healthy, Epilepsy, Depression, Parkinson's, Alzheimer's, Stroke, etc. |
| **Modality** | Visual, Auditory, Motor, Somatosensory, Resting State, Sleep, etc. |
| **Type** | Perception, Memory, Attention, Decision-making, Language, BCI, etc. |

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Remote Server                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Docker Container                       │   │
│  │                                                          │   │
│  │   ┌──────────────┐    ┌─────────────────────────────┐   │   │
│  │   │   FastAPI    │    │     eegdash-llm-tagger      │   │   │
│  │   │   (uvicorn)  │───▶│     (pip installed)         │   │   │
│  │   │   :8000      │    │                             │   │   │
│  │   └──────────────┘    └─────────────────────────────┘   │   │
│  │          │                                               │   │
│  │          ▼                                               │   │
│  │   ┌──────────────────────────────────────────────────┐  │   │
│  │   │  /tmp/eegdash-llm-api/cache/ (Docker Volume)     │  │   │
│  │   │  - tagging_cache.json                            │  │   │
│  │   │  - abstract_cache.json                           │  │   │
│  │   └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Environment Variables:                                         │
│  - OPENROUTER_API_KEY                                          │
│  - LLM_MODEL (optional)                                        │
│  - CACHE_DIR (optional)                                        │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/tag` | Tag a dataset (cache-first, LLM on miss) |
| `GET` | `/api/v1/tags/{dataset_id}` | Get cached tags only (no LLM call) |
| `GET` | `/api/v1/cache/stats` | Get cache statistics |
| `GET` | `/api/v1/cache/entries` | List cache entries |
| `DELETE` | `/api/v1/cache` | Clear entire cache |
| `DELETE` | `/api/v1/cache/{cache_key}` | Delete specific cache entry |
| `GET` | `/health` | Health check |

## Error Handling

```
┌─────────────────────────────────────────────────────────────┐
│                      Error Handling                         │
│                                                             │
│  Clone fails ──────▶ Return error response                  │
│                                                             │
│  Metadata extraction fails ──────▶ Return error response    │
│                                                             │
│  LLM call fails:                                            │
│  ├── serve_stale_on_error=true ──────▶ Return stale cache  │
│  └── serve_stale_on_error=false ─────▶ Return error        │
│                                                             │
│  Abstract fetch fails ──────▶ Continue without abstract     │
│                              (graceful degradation)         │
└─────────────────────────────────────────────────────────────┘
```

## Cost Optimization

1. **Content-addressable caching** - Same metadata + same config = same cache key = no duplicate LLM calls
2. **Metadata filtering** - Only relevant fields sent to LLM (reduces tokens)
3. **Abstract caching** - Paper abstracts cached to avoid repeated API calls
4. **Shallow cloning** - `git clone --depth 1` reduces clone time and bandwidth

## Security Considerations

1. **API Key Management** - `OPENROUTER_API_KEY` passed via environment variable, never in code
2. **Temporary Files** - Cloned repos stored in `/tmp`, auto-deleted after processing
3. **No Secrets in Cache** - Cache only stores classification results, not API keys
