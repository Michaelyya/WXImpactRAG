# BM25okapi, BM25Plus, BM25L. MiniLM for reranking.
import os
import pandas as pd
import numpy as np
import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from sentence_transformers import CrossEncoder
from constant.constants import FILE_QUERY_ADDRESS, FILE_CONCATENATED_CHUNKS_ADDRESS
from WeatherArchive_Retrieval.utils import BASE_ADDRESS


def bm25_retrieve_local_ids(df_queries, df_chunks, bm25_builder, top_k=100):
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    tokenized_corpus = [p.split() for p in passages]
    bm25 = bm25_builder(tokenized_corpus)

    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"BM25-{bm25_builder.__name__}",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        tokenized_query = query.split()

        scores = bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results[qid] = [ids[i] for i in top_idx]

    return results


def save_raw_results(results, df_queries, output_path, top_k=100):
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = row["query"]
        retrieved_ids = results.get(qid, [])
        row_data = [query] + retrieved_ids + [""] * (top_k - len(retrieved_ids))
        rows.append(row_data)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# Modified reranker (more robust + renamed)
def ce_rerank_ids(
    df_queries,
    df_chunks,
    bm25_results,
    ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=100,
    batch_size=128,
):
    from transformers import logging

    logging.set_verbosity_error()

    id2text = dict(zip(df_chunks["id"].astype(str), df_chunks["Text"].astype(str)))

    try:
        model = CrossEncoder(ce_model_name)
    except Exception as e:
        print(f"❌ Failed to load model: {ce_model_name}\n{e}")
        return {}

    results = {}

    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"CE rerank ({ce_model_name})",
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
                if hasattr(batch_scores, "tolist"):
                    scores.extend(batch_scores.tolist())
                else:
                    scores.extend(list(batch_scores))
            except Exception as e:
                print(f"❌ Model prediction failed (query: {qid}): {e}")
                scores.extend([0.0] * len(batch_pairs))

        order = np.argsort(scores)[::-1]
        reranked_ids = [candidate_ids[i] for i in order[:top_k]]
        results[qid] = reranked_ids

    return results


def run_and_eval_retrievers():
    TOP_K = 100
    CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in df_queries.columns:
        df_queries = df_queries.reset_index().rename(columns={"index": "id"})
    df_queries["id"] = df_queries["id"].astype(str)

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "id" not in df_chunks.columns:
        df_chunks = df_chunks.reset_index().rename(columns={"index": "id"})
    df_chunks["id"] = df_chunks["id"].astype(str)

    bm25_variants = [
        ("BM25Plus", BM25Plus),
        ("BM25L", BM25L),
        ("BM25Okapi", BM25Okapi),
    ]

    bm25_results_map = {}

    for name, builder in bm25_variants:
        res = bm25_retrieve_local_ids(df_queries, df_chunks, builder, top_k=TOP_K)
        bm25_results_map[name] = res
        out_raw = f"{BASE_ADDRESS}/raw_{name}_result.csv"
        save_raw_results(res, df_queries, out_raw, top_k=TOP_K)

    for name, _ in bm25_variants:
        ce_res = ce_rerank_ids(
            df_queries,
            df_chunks,
            bm25_results_map[name],
            ce_model_name=CE_MODEL,
            top_k=TOP_K,
        )
        out_ce = f"{BASE_ADDRESS}/raw_{name}_ce_reranked.csv"  # More consistent naming
        save_raw_results(ce_res, df_queries, out_ce, top_k=TOP_K)


if __name__ == "__main__":
    run_and_eval_retrievers()
