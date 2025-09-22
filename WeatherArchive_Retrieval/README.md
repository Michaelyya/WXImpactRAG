# Weather Archive Retrieval Evaluation

This module contains comprehensive retrieval evaluation scripts for weather archive data using various embedding models and retrieval methods.

## Retrieval Evaluation Scripts

### retriever_eval_1.py - BM25 Variants with Cross-Encoder Reranking

Evaluates BM25 variants (BM25Okapi, BM25Plus, BM25L) with MiniLM cross-encoder reranking.

**Models tested:**

- BM25Okapi
- BM25Plus
- BM25L
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_1
```

### retriever_eval_2.py - SentenceTransformer Models

Evaluates dense retrieval using SentenceTransformer models with FAISS indexing.

**Models tested:**

- SBERT (`sentence-transformers/msmarco-distilbert-base-tas-b`)
- SPLADE (`naver/splade-cocondenser-ensembledistil`)

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_2
```

### retriever_eval_3.py - ANCE and UniCoil

Evaluates dense retrieval using ANCE and UniCoil models.

**Models tested:**

- ANCE (`castorini/ance-msmarco-passage`)
- UniCoil (`castorini/unicoil-msmarco-passage`)

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_3
```

### retriever_eval_4.py - Qwen Embedding Models

Evaluates Qwen embedding models using both SentenceTransformer and transformers approaches.

**Models tested:**

- Qwen3-Embedding-0.6B (`Qwen/Qwen3-Embedding-0.6B`)
- Qwen3-Embedding-4B (`Qwen/Qwen3-Embedding-4B`) - commented out
- Qwen3-Embedding-8B (`Qwen/Qwen3-Embedding-8B`) - commented out

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_4
```

### retriever_eval_5.py - OpenAI Embedding Models

Evaluates OpenAI embedding models using their API.

**Models tested:**

- text-embedding-3-large
- text-embedding-3-small
- text-embedding-ada-002

**Requirements:**

- OpenAI API key in environment variable `OPENAI_API_KEY`

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_5
```

### retriever_eval_6.py - Arctic and Granite Models

Evaluates Arctic and Granite embedding models.

**Models tested:**

- Arctic Embed 2.0 (`Snowflake/snowflake-arctic-embed-l-v2.0`)
- Granite Embedding R2 (`ibm-granite/granite-embedding-english-r2`)

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_6
```

### retriever_eval_7.py - Google Gemini Embedding

Evaluates Google Gemini embedding models with multi-threading support.

**Models tested:**

- Gemini Embedding 001 (`models/gemini-embedding-001`)
- Text Embedding 004 (`models/text-embedding-004`) - fallback

**Features:**

- Multi-threading support (15 threads)
- Rate limiting (300 RPM)
- Automatic model fallback

**Requirements:**

- Google API key in environment variable `GOOGLE_API_KEY`

**Run:**

```bash
python -m WeatherArchive_Retrieval.retriever_eval_7
```

## Get Overall Statistical Results

To get comprehensive evaluation results across all models:

```bash
python -m WeatherArchive_Retrieval.overall
```

## Output Format

All scripts generate raw CSV files in the `output/` directory with the following format:

- **Columns:** `query`, `top_1`, `top_2`, ..., `top_100`
- **Content:** Document IDs for each query's top-100 retrieved results
- **Naming:** `raw_[model_name]_result.csv` or `raw_[bm25_variant]_ce_reranked.csv`

## Requirements

- Python 3.7+
- Required packages: pandas, numpy, tqdm, sentence-transformers, faiss, rank-bm25, transformers, torch
- For OpenAI models: `openai` package and API key
- For Gemini models: `google-generativeai` package and API key
- For Qwen models: Sufficient GPU memory for large models

## Data Requirements

- `queries.csv`: Must contain `query` column
- `concatenated_chunks.csv`: Must contain `Text` column
- Both files will auto-generate `id` columns if not present
