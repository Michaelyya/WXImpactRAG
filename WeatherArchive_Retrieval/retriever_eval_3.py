# Ance, unicoil
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tqdm

from WeatherArchive_Retrieval.utils import BASE_ADDRESS

# Only need queries and chunks files
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/query.csv"  needs column: 'query' (can have no 'id')
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv"  needs column: 'Text' (can have no 'id')
)


# -----------------------------
# Utils: Ensure string type id column exists
# -----------------------------
def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


# -----------------------------
# Dense retrieval: return top_k **document ids**
# -----------------------------
def dense_retrieve_local_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str,
    top_k: int = 100,
):
    """
    Use SentenceTransformer + FAISS for vector retrieval.
    Returns: {qid(str): [doc_id(str), ...]} (length <= top_k)
    """
    passages = df_chunks["Text"].astype(str).tolist()
    doc_ids = df_chunks["id"].astype(str).tolist()
    if len(passages) == 0:
        raise ValueError(
            "Passages is empty, please check 'Text' column in concatenated_chunks.csv."
        )

    print(f"[{model_name}] Loading model ...")
    model = SentenceTransformer(model_name)

    print(f"[{model_name}] Encoding passages ...")
    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = passage_embeddings.shape[1]

    # Normalized vectors + inner product index = cosine similarity ranking
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    results = {}
    print(f"[{model_name}] Retrieving ...")
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])

        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        k = min(top_k, len(passages))
        scores, indices = index.search(q_emb, k)

        top_idx = indices[0] if isinstance(indices, np.ndarray) else indices
        results[qid] = [doc_ids[i] for i in top_idx]

    return results


# -----------------------------
# Save raw CSV: query + top_1..top_k (doc_id)
# -----------------------------
def save_raw_results_as_csv(
    results: dict,
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
):
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        ids = results.get(qid, [])
        ids = (ids + [""] * top_k)[
            :top_k
        ]  # Pad with empty if insufficient, truncate if exceeds
        rows.append([query] + ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


# -----------------------------
# Main process: run multiple retrievers and export raw CSV for each
# -----------------------------
def WeatherArchive_Retrieval_and_dump_raw():
    # Read queries
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv missing 'query' column.")
    df_queries = ensure_id_column(df_queries, "id")

    # Read chunks (ensure has id)
    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv missing 'Text' column.")
    df_chunks = ensure_id_column(df_chunks, "id")

    # Three models you provided (if unavailable, will be skipped by try/except)
    retrievers = [
        ("ance", "castorini/ance-msmarco-passage"),
        ("unicoil", "castorini/unicoil-msmarco-passage"),
    ]

    for short_name, model_name in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model_name}) ===")
        try:
            results = dense_retrieve_local_ids(
                df_queries=df_queries,
                df_chunks=df_chunks,
                model_name=model_name,
                top_k=100,
            )
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{short_name}' failed: {e}")


if __name__ == "__main__":
    WeatherArchive_Retrieval_and_dump_raw()
