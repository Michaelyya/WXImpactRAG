# Weather Archive Retrieval Evaluation — Pyserini Implementation

This package is a [Pyserini](https://github.com/castorini/pyserini)-backed
re-implementation of `WeatherArchive_Retrieval`. It keeps the **same inputs,
the same per-model output files, and the same `overall.py` metrics**, but routes
every retriever through Pyserini's index / search abstractions instead of the
original `rank_bm25` + raw-FAISS code.

| Concern | `WeatherArchive_Retrieval` (original) | `WeatherArchive_Retrieval_pyserini` (this package) |
|---|---|---|
| Sparse BM25 | `rank_bm25` (in-memory) | `pyserini.index.lucene` + `LuceneSearcher` |
| Learned sparse (SPLADE, uniCOIL) | treated as dense / dense | `pyserini.encode` (impact) + `LuceneImpactSearcher` |
| Dense (SBERT, ANCE, Qwen, Arctic, Granite) | SentenceTransformer + `faiss.IndexFlatIP` | `pyserini.encode --to-faiss` + `FaissSearcher` |
| API embeddings (OpenAI, Gemini) | API → `faiss.IndexFlatIP` | API → `pyserini.index.faiss` + `FaissSearcher` (custom query encoder) |

Inputs, outputs, and `id` conventions are unchanged, so the result files are
drop-in compatible with the original `overall.py` aggregation.

## Inputs / Outputs

* **Inputs** (from `constant/constants.py`, shared with the original package):
  * `queries.csv` — must contain a `query` column (an `id` column is generated
    from the row index if absent).
  * `concatenated_chunks.csv` — must contain a `Text` column (likewise auto-`id`).
* **Outputs** — `retriever_eval/raw_*.csv` with columns `query, top_1, …, top_100`
  (values are document ids), identical schema to the original.
* **Cached artifacts** — the JSONL collection (`collections/`) and every Pyserini
  index (`indexes/`) are built once and reused. Delete those folders to rebuild.

## Model → Pyserini component map

| Script | Model(s) | Pyserini path | Output file |
|---|---|---|---|
| `retriever_eval_1.py` | BM25Okapi | `LuceneSearcher` (Okapi BM25, k1=1.5, b=0.75) | `raw_BM25Okapi_result.csv` |
| | BM25Plus, BM25L | **`rank_bm25`** (Lucene has no BM25+/BM25L) | `raw_BM25Plus_result.csv`, `raw_BM25L_result.csv` |
| | + cross-encoder rerank (all 3) | `sentence-transformers` CrossEncoder | `raw_BM25*_ce_reranked.csv` |
| `retriever_eval_2.py` | SBERT (`msmarco-distilbert-base-tas-b`) | `FaissSearcher` + `AutoQueryEncoder` (CLS) | `raw_model_result_sbert.csv` |
| | SPLADE (`splade-cocondenser-ensembledistil`) | `LuceneImpactSearcher` (impact) | `raw_model_result_splade.csv` |
| `retriever_eval_3.py` | ANCE (`ance-msmarco-passage`) | `FaissSearcher` + `AnceQueryEncoder` | `raw_model_result_ance.csv` |
| | uniCOIL (`unicoil-msmarco-passage`) | `LuceneImpactSearcher` (impact) | `raw_model_result_unicoil.csv` |
| `retriever_eval_4.py` | Qwen3-Embedding-0.6B | `FaissSearcher` + `AutoQueryEncoder` (CLS) | `raw_model_result_qwen3-0_6b.csv` |
| `retriever_eval_5.py` | OpenAI 3-large / 3-small / ada-002 | `pyserini.index.faiss` + `FaissSearcher` | `raw_model_result_openai-*.csv` |
| `retriever_eval_6.py` | Arctic Embed 2.0 (CLS), Granite R2 (mean) | `FaissSearcher` + `AutoQueryEncoder` | `raw_model_result_arctic.csv`, `…granite.csv` |
| `retriever_eval_7.py` | Gemini Embedding 001 | `pyserini.index.faiss` + `FaissSearcher` | `raw_model_result_gemini-embedding-001.csv` |

### Design notes

* **Hybrid BM25.** Lucene (and therefore Pyserini) implements only Okapi BM25.
  The original benchmarks three `rank_bm25` variants, so BM25Okapi is served by
  Pyserini's `LuceneSearcher` (k1=1.5, b=0.75 to match `rank_bm25`'s Okapi
  defaults) while BM25Plus / BM25L stay on `rank_bm25` to remain faithful.
* **SPLADE / uniCOIL** are genuine learned-sparse models; the original code ran
  them through a dense path, but here they use Pyserini's impact pipeline
  (`pyserini.encode` → `JsonVectorCollection` + `--impact` → `LuceneImpactSearcher`),
  which is the correct treatment.
* **Dense pooling** is set per model (CLS for tas-b / Qwen / Arctic, mean for
  Granite) and applied identically on the document and query sides so the index
  stays consistent. Embeddings are L2-normalised, matching the original's
  `normalize_embeddings=True` + inner-product index.
* **OpenAI / Gemini.** Pyserini has no portable cross-version encoder for these,
  so vectors are produced with the original API code (the Gemini path keeps the
  multi-threaded, rate-limited embedder) and handed to Pyserini via
  `pyserini.index.faiss` + a tiny custom query encoder exposing `encode()`.

## Running

```bash
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_1   # BM25 variants + CE rerank
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_2   # SBERT, SPLADE
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_3   # ANCE, uniCOIL
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_4   # Qwen
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_5   # OpenAI (needs OPENAI_API_KEY)
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_6   # Arctic, Granite
python -m WeatherArchive_Retrieval_pyserini.retriever_eval_7   # Gemini (needs GOOGLE_API_KEY)

python -m WeatherArchive_Retrieval_pyserini.overall            # aggregate Recall/nDCG/MRR/BLEU
```

## Requirements

* `pyserini` (already pinned in the repo `requirements.txt` as `pyserini==1.2.0`).
* **A Java runtime (JDK 11+)** — Pyserini's Lucene backend runs on the JVM via
  `pyjnius`. This is the main extra dependency over the original package.
* `faiss`, `sentence-transformers`, `transformers`, `torch`, `rank-bm25`,
  `pandas`, `numpy`, `tqdm`, `nltk` (BLEU in `overall.py`).
* `openai` + `OPENAI_API_KEY` for `retriever_eval_5`;
  `google-generativeai` + `GOOGLE_API_KEY` for `retriever_eval_7`.
* The full corpus `concatenated_chunks.csv` (Git LFS, ~1.3 GB) must be present.

> **Pyserini version note.** Pyserini occasionally renames CLI flags and encoder
> classes between releases. All version-sensitive commands (index/encode CLI
> invocations) and encoder constructions live in `pyserini_utils.py`, so if the
> installed build differs from `pyserini==1.2.0` they can be adjusted in one
> place. The encoder factories already try multiple known import paths and fall
> back gracefully.
