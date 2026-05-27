# -*- coding: utf-8 -*-
"""
retriever_eval_1 (Pyserini) -- BM25 variants + MiniLM cross-encoder reranking.

Hybrid BM25, as requested:
    * BM25Okapi -> Pyserini Lucene BM25 (Okapi; k1=1.5, b=0.75 to match
      rank_bm25's BM25Okapi defaults). This is the Pyserini-backed variant.
    * BM25Plus / BM25L -> kept on rank_bm25, because Lucene does not implement
      the BM25+ / BM25L scoring functions, so reproducing them faithfully
      requires the original library.

Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2) is applied to all
three variants, identical to WeatherArchive_Retrieval/retriever_eval_1.py.

Outputs (consumed by overall.py):
    raw_BM25Okapi_result.csv, raw_BM25Plus_result.csv, raw_BM25L_result.csv
    raw_BM25Okapi_ce_reranked.csv, raw_BM25Plus_ce_reranked.csv,
    raw_BM25L_ce_reranked.csv
"""

import os

import numpy as np
import tqdm
from rank_bm25 import BM25L, BM25Plus
from sentence_transformers import CrossEncoder

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    run_bm25_pyserini,
    save_raw_results_as_csv,
)

TOP_K = 100
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rank_bm25_retrieve_local_ids(df_queries, df_chunks, bm25_builder, top_k=TOP_K):
    """rank_bm25 retrieval (used for BM25Plus / BM25L, which Lucene lacks)."""
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    tokenized_corpus = [p.split() for p in passages]
    bm25 = bm25_builder(tokenized_corpus)

    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(), total=len(df_queries), desc=f"BM25-{bm25_builder.__name__}"
    ):
        qid = str(row["id"])
        tokenized_query = str(row["query"]).split()
        scores = bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results[qid] = [ids[i] for i in top_idx]
    return results


def ce_rerank_ids(
    df_queries,
    df_chunks,
    bm25_results,
    ce_model_name=CE_MODEL,
    top_k=TOP_K,
    batch_size=128,
):
    """Rerank each query's BM25 candidate ids with a cross-encoder."""
    from transformers import logging

    logging.set_verbosity_error()

    id2text = dict(zip(df_chunks["id"].astype(str), df_chunks["Text"].astype(str)))

    try:
        model = CrossEncoder(ce_model_name)
    except Exception as e:
        print(f"Failed to load cross-encoder: {ce_model_name}\n{e}")
        return {}

    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(), total=len(df_queries), desc=f"CE rerank ({ce_model_name})"
    ):
        qid = str(row["id"])
        query = str(row["query"])

        candidate_ids = bm25_results.get(qid, [])
        if not candidate_ids:
            results[qid] = []
            continue

        pairs = [(query, id2text[cid]) for cid in candidate_ids if cid in id2text]
        if not pairs:
            results[qid] = []
            continue

        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            try:
                batch_scores = model.predict(batch_pairs)
                scores.extend(
                    batch_scores.tolist()
                    if hasattr(batch_scores, "tolist")
                    else list(batch_scores)
                )
            except Exception as e:
                print(f"Cross-encoder prediction failed (query {qid}): {e}")
                scores.extend([0.0] * len(batch_pairs))

        order = np.argsort(scores)[::-1]
        results[qid] = [candidate_ids[i] for i in order[:top_k]]
    return results


def run_and_eval_retrievers():
    df_queries = load_queries()
    df_chunks = load_chunks()

    # --- BM25 retrieval (hybrid Pyserini + rank_bm25) ---
    bm25_results_map = {}

    # Okapi via Pyserini Lucene.
    bm25_results_map["BM25Okapi"] = run_bm25_pyserini(
        df_queries, df_chunks, k1=1.5, b=0.75, top_k=TOP_K
    )

    # Plus / L via rank_bm25 (no Lucene equivalent).
    for name, builder in [("BM25Plus", BM25Plus), ("BM25L", BM25L)]:
        bm25_results_map[name] = rank_bm25_retrieve_local_ids(
            df_queries, df_chunks, builder, top_k=TOP_K
        )

    for name, res in bm25_results_map.items():
        save_raw_results_as_csv(
            res, df_queries, f"{BASE_ADDRESS}/raw_{name}_result.csv", top_k=TOP_K
        )

    # --- Cross-encoder reranking for every variant ---
    for name in ["BM25Okapi", "BM25Plus", "BM25L"]:
        ce_res = ce_rerank_ids(
            df_queries, df_chunks, bm25_results_map[name], ce_model_name=CE_MODEL, top_k=TOP_K
        )
        save_raw_results_as_csv(
            ce_res, df_queries, f"{BASE_ADDRESS}/raw_{name}_ce_reranked.csv", top_k=TOP_K
        )


if __name__ == "__main__":
    os.makedirs(BASE_ADDRESS, exist_ok=True)
    run_and_eval_retrievers()
