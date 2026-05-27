# -*- coding: utf-8 -*-
"""
Shared helpers for the Pyserini re-implementation of WeatherArchive-Retrieval.

The original WeatherArchive_Retrieval package builds an ad-hoc index in memory
(rank_bm25 for sparse, SentenceTransformer + raw FAISS for dense) every time a
script runs.  This package keeps the *exact same inputs and outputs* but routes
every retriever through Pyserini's index/search abstractions:

    * sparse BM25            -> pyserini.index.lucene  + LuceneSearcher
    * learned sparse (SPLADE,
      uniCOIL)               -> pyserini.encode (impact) + LuceneImpactSearcher
    * dense (SBERT, ANCE,
      Qwen, Arctic, Granite) -> pyserini.encode --to-faiss + FaissSearcher
    * API embeddings (OpenAI,
      Gemini)                -> custom embeddings + pyserini.index.faiss
                                + FaissSearcher with a custom QueryEncoder

Inputs (identical to the original package, taken from constant.constants):
    queries.csv               -> column 'query' (an 'id' column is generated if
                                 absent, using the row index as a string)
    concatenated_chunks.csv   -> column 'Text' (likewise generates 'id')

Output (identical to the original package, consumed by overall.py):
    raw_*.csv with columns: query, top_1, ..., top_100   (values are doc ids)

Heavy artifacts (the JSONL collection and every Pyserini index) are cached under
``indexes/`` and ``collections/`` and reused on subsequent runs.

NOTE ON PYSERINI VERSIONS: Pyserini occasionally renames CLI flags / encoder
classes between releases.  Every version-sensitive command is constructed in
this single module so it can be adjusted in one place if the installed build
differs from the one targeted here (pyserini==1.2.0, see requirements.txt).
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import pandas as pd
import tqdm

from constant.constants import (
    FILE_QUERY_ADDRESS,
    FILE_CONCATENATED_CHUNKS_ADDRESS,
    FILE_CORRECT_PASSAGES_ADDRESS,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = "WeatherArchive_Retrieval_pyserini"

# Where raw_*.csv result files are written (mirrors WeatherArchive_Retrieval).
BASE_ADDRESS = os.path.join(PACKAGE_DIR, "retriever_eval")

# Cached Pyserini collections / indexes.
COLLECTION_BASE = os.path.join(PACKAGE_DIR, "collections")
INDEX_BASE = os.path.join(PACKAGE_DIR, "indexes")

TOP_K_DEFAULT = 100


# --------------------------------------------------------------------------- #
# Small data utilities (kept byte-for-byte compatible with the original logic)
# --------------------------------------------------------------------------- #
def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Ensure a unique string ``id`` column exists; generate from the row index."""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def load_queries(path: str = FILE_QUERY_ADDRESS) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("queries.csv needs to contain a 'query' column.")
    return ensure_id_column(df, "id")


def load_chunks(path: str = FILE_CONCATENATED_CHUNKS_ADDRESS) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Text" not in df.columns:
        raise ValueError("concatenated_chunks.csv needs to contain a 'Text' column.")
    return ensure_id_column(df, "id")


def pick_device() -> str:
    """Return 'cuda:0' when a GPU is available, else 'cpu' (mirrors ST autodetect)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Collection preparation (CSV -> Pyserini JSONL)
# --------------------------------------------------------------------------- #
def _dir_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def prepare_collection(
    df_chunks: pd.DataFrame,
    name: str = "corpus",
    text_col: str = "Text",
    id_col: str = "id",
) -> str:
    """
    Write ``df_chunks`` to a Pyserini ``JsonCollection`` directory.

    Each line is ``{"id": <id>, "contents": <Text>}``.  The directory is cached
    and reused; delete it to force a rebuild.  Returns the collection directory.
    """
    collection_dir = os.path.join(COLLECTION_BASE, name)
    jsonl_path = os.path.join(collection_dir, "docs.jsonl")

    if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        print(f"[collection] reusing cached collection: {jsonl_path}")
        return collection_dir

    os.makedirs(collection_dir, exist_ok=True)
    ids = df_chunks[id_col].astype(str).tolist()
    texts = df_chunks[text_col].astype(str).tolist()
    if not texts:
        raise ValueError("Corpus is empty: no usable 'Text' found.")

    print(f"[collection] writing {len(texts)} docs -> {jsonl_path}")
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for doc_id, text in tqdm.tqdm(
            zip(ids, texts), total=len(ids), desc="Writing JSONL"
        ):
            fout.write(
                json.dumps({"id": doc_id, "contents": text}, ensure_ascii=False) + "\n"
            )
    return collection_dir


# --------------------------------------------------------------------------- #
# Subprocess wrapper
# --------------------------------------------------------------------------- #
def _run(cmd: List[str]) -> None:
    """Run a Pyserini CLI command, streaming output; raise on non-zero exit."""
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Index builders (Pyserini CLI)
# --------------------------------------------------------------------------- #
def build_lucene_index(
    input_dir: str,
    index_dir: str,
    impact: bool = False,
    threads: int = 8,
) -> str:
    """
    Build a Lucene index with ``pyserini.index.lucene``.

    impact=False -> JsonCollection (BM25 bag-of-words index).
    impact=True  -> JsonVectorCollection of pre-tokenized term weights, indexed
                    with --impact (used for SPLADE / uniCOIL learned sparse).
    """
    if _dir_has_files(index_dir):
        print(f"[lucene] reusing cached index: {index_dir}")
        return index_dir

    os.makedirs(index_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonVectorCollection" if impact else "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
    ]
    if impact:
        cmd += ["--impact", "--pretokenized"]
    else:
        cmd += ["--storePositions", "--storeDocvectors", "--storeRaw"]
    _run(cmd)
    return index_dir


def encode_corpus_to_faiss(
    corpus_jsonl: str,
    index_dir: str,
    encoder: str,
    encoder_class: str = "auto",
    pooling: str = "cls",
    l2_norm: bool = True,
    device: Optional[str] = None,
    batch: int = 64,
    fp16: bool = True,
    fields: str = "contents",
) -> str:
    """
    Encode the corpus into a Pyserini FAISS index with ``pyserini.encode``.

    Mirrors the original dense pipeline (SentenceTransformer.encode with
    normalize_embeddings=True -> IndexFlatIP), but Pyserini owns the encoding
    and index format so it can be queried with FaissSearcher.
    """
    if _dir_has_files(index_dir):
        print(f"[faiss] reusing cached index: {index_dir}")
        return index_dir

    os.makedirs(index_dir, exist_ok=True)
    device = device or pick_device()
    cmd = [
        sys.executable, "-m", "pyserini.encode",
        "input", "--corpus", corpus_jsonl, "--fields", fields,
        "output", "--embeddings", index_dir, "--to-faiss",
        "encoder", "--encoder", encoder, "--encoder-class", encoder_class,
        "--fields", fields, "--batch", str(batch),
        "--pooling", pooling, "--device", device,
    ]
    if l2_norm:
        cmd += ["--l2-norm"]
    if fp16 and device != "cpu":
        cmd += ["--fp16"]
    _run(cmd)
    return index_dir


def encode_corpus_to_impact(
    corpus_jsonl: str,
    vectors_dir: str,
    encoder: str,
    encoder_class: str,
    device: Optional[str] = None,
    batch: int = 32,
    fields: str = "contents",
    weight_range: Optional[int] = None,
    quant_range: Optional[int] = None,
) -> str:
    """
    Encode the corpus into JSONL impact vectors (term -> integer weight) with
    ``pyserini.encode``.  The output directory is then indexed with
    build_lucene_index(impact=True).  Used for SPLADE / uniCOIL.
    """
    if _dir_has_files(vectors_dir):
        print(f"[impact] reusing cached vectors: {vectors_dir}")
        return vectors_dir

    os.makedirs(vectors_dir, exist_ok=True)
    device = device or pick_device()
    cmd = [
        sys.executable, "-m", "pyserini.encode",
        "input", "--corpus", corpus_jsonl, "--fields", fields,
        "output", "--embeddings", vectors_dir,
        "encoder", "--encoder", encoder, "--encoder-class", encoder_class,
        "--fields", fields, "--batch", str(batch), "--device", device,
    ]
    if weight_range is not None:
        cmd += ["--weight-range", str(weight_range)]
    if quant_range is not None:
        cmd += ["--quant-range", str(quant_range)]
    _run(cmd)
    return vectors_dir


def build_faiss_index_from_vectors(vectors_dir: str, index_dir: str) -> str:
    """
    Build a FAISS flat index from a directory of pre-computed embedding JSONL
    files ({"id": ..., "vector": [...]}) with ``pyserini.index.faiss``.

    Used for API embedders (OpenAI, Gemini) whose vectors are computed outside
    Pyserini and then handed to FaissSearcher.
    """
    if _dir_has_files(index_dir):
        print(f"[faiss] reusing cached index: {index_dir}")
        return index_dir

    os.makedirs(index_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.faiss",
        "--input", vectors_dir,
        "--output", index_dir,
    ]
    _run(cmd)
    return index_dir


def write_embedding_jsonl(
    ids: List[str],
    vectors,
    vectors_dir: str,
    filename: str = "embeddings.jsonl",
) -> str:
    """Write {"id", "vector"} JSONL for build_faiss_index_from_vectors()."""
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, filename)
    with open(path, "w", encoding="utf-8") as fout:
        for doc_id, vec in zip(ids, vectors):
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            fout.write(json.dumps({"id": str(doc_id), "vector": vec_list}) + "\n")
    return vectors_dir


# --------------------------------------------------------------------------- #
# Search + result serialization
# --------------------------------------------------------------------------- #
def search_to_results(
    searcher,
    df_queries: pd.DataFrame,
    top_k: int = TOP_K_DEFAULT,
    desc: str = "Retrieving",
) -> Dict[str, List[str]]:
    """
    Run every query through a Pyserini searcher and collect top-k doc ids.

    Works for LuceneSearcher, LuceneImpactSearcher and FaissSearcher -- every
    hit object exposes ``.docid``.  Returns {qid(str): [doc_id(str), ...]}.
    """
    results: Dict[str, List[str]] = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(), total=len(df_queries), desc=desc
    ):
        qid = str(row["id"])
        query = str(row["query"])
        try:
            hits = searcher.search(query, k=top_k)
            results[qid] = [h.docid for h in hits]
        except Exception as e:  # e.g. Lucene query-parser edge cases
            print(f"[warn] search failed for qid={qid}: {e}")
            results[qid] = []
    return results


def save_raw_results_as_csv(
    results: Dict[str, List[str]],
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = TOP_K_DEFAULT,
) -> None:
    """
    Save {qid: [doc_id, ...]} as raw CSV (query, top_1..top_k), aligned with
    df_queries order -- identical schema to WeatherArchive_Retrieval.
    """
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        retrieved_ids = results.get(qid, [])
        retrieved_ids = (retrieved_ids + [""] * top_k)[:top_k]
        rows.append([query] + retrieved_ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


# --------------------------------------------------------------------------- #
# Query-encoder factories
#
# Encoder class names / locations occasionally move between Pyserini releases,
# so every construction is funnelled through these factories and guarded with
# fallbacks.  Adjust here (one place) if the installed build differs.
# --------------------------------------------------------------------------- #
def make_auto_query_encoder(
    model: str,
    device: Optional[str] = None,
    pooling: str = "cls",
    l2_norm: bool = True,
):
    """Generic HF dense query encoder (SBERT / Qwen / Arctic / Granite)."""
    from pyserini.search.faiss import AutoQueryEncoder

    return AutoQueryEncoder(
        encoder_dir=model,
        device=device or pick_device(),
        pooling=pooling,
        l2_norm=l2_norm,
    )


def make_ance_query_encoder(model: str, device: Optional[str] = None):
    """Native ANCE query encoder."""
    from pyserini.search.faiss import AnceQueryEncoder

    return AnceQueryEncoder(encoder_dir=model, device=device or pick_device())


def make_impact_query_encoder(kind: str, model: str, device: Optional[str] = None):
    """
    Learned-sparse query encoder for LuceneImpactSearcher.

    kind in {'splade', 'unicoil'}.  Tries the known import paths/class names
    across Pyserini versions and finally falls back to handing the raw model
    name to LuceneImpactSearcher (which can build the encoder itself).
    """
    device = device or pick_device()
    attempts = []
    if kind == "splade":
        def _a():
            from pyserini.encode import SpladeQueryEncoder
            return SpladeQueryEncoder(model, device=device)

        def _b():
            from pyserini.search.lucene import SpladeQueryEncoder
            return SpladeQueryEncoder(model, device=device)

        attempts = [_a, _b]
    elif kind == "unicoil":
        def _a():
            from pyserini.encode import UniCoilQueryEncoder
            return UniCoilQueryEncoder(model, device=device)

        def _b():
            from pyserini.search.lucene import UniCoilQueryEncoder
            return UniCoilQueryEncoder(model, device=device)

        attempts = [_a, _b]
    else:
        raise ValueError(f"Unknown impact encoder kind: {kind}")

    last_err = None
    for build in attempts:
        try:
            return build()
        except Exception as e:  # ImportError or signature mismatch
            last_err = e
    print(
        f"[warn] could not build a {kind} query-encoder object ({last_err}); "
        f"falling back to passing the model name to LuceneImpactSearcher."
    )
    return model


# --------------------------------------------------------------------------- #
# High-level retrieval pipelines
# --------------------------------------------------------------------------- #
def run_bm25_pyserini(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, List[str]]:
    """
    Okapi BM25 via Pyserini's Lucene backend.

    Lucene implements Okapi BM25; (k1, b) default to (1.5, 0.75) to match
    rank_bm25's BM25Okapi defaults used by the original package.
    """
    from pyserini.search.lucene import LuceneSearcher

    collection_dir = prepare_collection(df_chunks)
    index_dir = build_lucene_index(
        collection_dir, os.path.join(INDEX_BASE, "lucene_bm25"), impact=False
    )
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1, b)
    return search_to_results(searcher, df_queries, top_k, desc="BM25 (Lucene/Okapi)")


def run_dense_faiss(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    short_name: str,
    encoder_model: str,
    query_encoder,
    encoder_class: str = "auto",
    pooling: str = "cls",
    l2_norm: bool = True,
    device: Optional[str] = None,
    batch: int = 64,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, List[str]]:
    """Encode the corpus into a Pyserini FAISS index and search it."""
    from pyserini.search.faiss import FaissSearcher

    device = device or pick_device()
    collection_dir = prepare_collection(df_chunks)
    corpus_jsonl = os.path.join(collection_dir, "docs.jsonl")
    index_dir = encode_corpus_to_faiss(
        corpus_jsonl,
        os.path.join(INDEX_BASE, f"faiss_{short_name}"),
        encoder=encoder_model,
        encoder_class=encoder_class,
        pooling=pooling,
        l2_norm=l2_norm,
        device=device,
        batch=batch,
    )
    searcher = FaissSearcher(index_dir, query_encoder)
    return search_to_results(searcher, df_queries, top_k, desc=f"Retrieving {short_name}")


def run_impact_lucene(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    short_name: str,
    encoder_model: str,
    encoder_class: str,
    query_encoder,
    weight_range: Optional[int] = None,
    quant_range: Optional[int] = None,
    device: Optional[str] = None,
    batch: int = 32,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, List[str]]:
    """Encode the corpus into impact vectors, build a Lucene impact index, search."""
    from pyserini.search.lucene import LuceneImpactSearcher

    device = device or pick_device()
    collection_dir = prepare_collection(df_chunks)
    corpus_jsonl = os.path.join(collection_dir, "docs.jsonl")
    vectors_dir = encode_corpus_to_impact(
        corpus_jsonl,
        os.path.join(INDEX_BASE, f"impact_vectors_{short_name}"),
        encoder=encoder_model,
        encoder_class=encoder_class,
        device=device,
        batch=batch,
        weight_range=weight_range,
        quant_range=quant_range,
    )
    index_dir = build_lucene_index(
        vectors_dir, os.path.join(INDEX_BASE, f"impact_index_{short_name}"), impact=True
    )
    searcher = LuceneImpactSearcher(index_dir, query_encoder)
    return search_to_results(searcher, df_queries, top_k, desc=f"Retrieving {short_name}")


def run_dense_precomputed(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    short_name: str,
    doc_vectors,
    query_encoder,
    top_k: int = TOP_K_DEFAULT,
) -> Dict[str, List[str]]:
    """
    Build a Pyserini FAISS index from externally-computed document vectors
    (OpenAI / Gemini) and search it with a custom query encoder.

    ``doc_vectors`` is an array aligned with df_chunks order; ``query_encoder``
    is any object exposing ``encode(query: str) -> np.ndarray``.
    """
    from pyserini.search.faiss import FaissSearcher

    ids = df_chunks["id"].astype(str).tolist()
    vectors_dir = write_embedding_jsonl(
        ids, doc_vectors, os.path.join(INDEX_BASE, f"api_vectors_{short_name}")
    )
    index_dir = build_faiss_index_from_vectors(
        vectors_dir, os.path.join(INDEX_BASE, f"api_index_{short_name}")
    )
    searcher = FaissSearcher(index_dir, query_encoder)
    return search_to_results(searcher, df_queries, top_k, desc=f"Retrieving {short_name}")


# --------------------------------------------------------------------------- #
# Per-query evaluation (parity with WeatherArchive_Retrieval/utils.py)
# --------------------------------------------------------------------------- #
def evaluate_retriever_performance(
    retrieval_results: Dict[str, List[str]],
    output_path: str,
    ks: List[int] = (1, 5, 10, 50, 100),
):
    """
    Recall@k / MRR@k against the gold passage in correct_passages.csv (one row
    per query, aligned with queries.csv).  ``retrieval_results`` maps qid to a
    list of *passage texts*.  Kept identical to the original utils.py.
    """
    qdf = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in qdf.columns:
        qdf = qdf.reset_index().rename(columns={"index": "id"})
    qdf["id"] = qdf["id"].astype(str)

    gdf = pd.read_csv(FILE_CORRECT_PASSAGES_ADDRESS)
    if "correct_passage" not in gdf.columns or len(gdf) != len(qdf):
        raise ValueError(
            "correct_passages.csv must contain 'correct_passage' column, and row "
            "count must align with query.csv."
        )

    qdf = qdf.copy()
    qdf["correct_passage"] = gdf["correct_passage"].astype(str)

    results = []
    ks = list(sorted(set(int(k) for k in ks)))

    for _, row in qdf.iterrows():
        qid = row["id"]
        query = str(row["query"])
        golden_answer = str(row["correct_passage"])

        retrieved_passages = retrieval_results.get(qid, [])

        hit_k = {f"recall@{k}": 0 for k in ks}
        mrr_k = {f"mrr@{k}": 0.0 for k in ks}
        hit_rank = -1

        max_k = max(ks) if ks else len(retrieved_passages)
        for rank, p in enumerate(retrieved_passages[:max_k], start=1):
            if str(p).strip() == golden_answer.strip():
                hit_rank = rank
                for k in ks:
                    if rank <= k:
                        hit_k[f"recall@{k}"] = 1
                        mrr_k[f"mrr@{k}"] = 1.0 / rank
                break

        results.append(
            {"id": qid, "query": query, "hit_rank": hit_rank, **hit_k, **mrr_k}
        )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
