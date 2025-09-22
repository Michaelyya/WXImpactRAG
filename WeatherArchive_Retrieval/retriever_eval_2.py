# splade, sbert
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tqdm

# Constants: Your project should already have these paths; if not, you can directly replace with string paths
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/queries.csv", needs 'query' column
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv", needs 'Text' column
)
from WeatherArchive_Retrieval.utils import (
    BASE_ADDRESS,
)  # e.g., "WeatherArchive_Retrieval/retriever_eval"


def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    Ensure DataFrame has a unique id column; if not exists, generate using row numbers (string).
    """
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def dense_retrieve_local_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str,
    top_k: int = 100,
):
    """
    Use SentenceTransformer + FAISS for dense retrieval, return top_k document **id list** for each query.
    - df_queries: needs columns ['id', 'query']
    - df_chunks: needs columns ['id', 'Text']
    Returns: {qid(str): [doc_id(str), ...]}
    """
    # Extract text and id, encode corpus
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if len(passages) == 0:
        raise ValueError(
            "Corpus is empty: no usable 'Text' found in concatenated_chunks.csv."
        )

    print(f"[{model_name}] Loading model...")
    model = SentenceTransformer(model_name)

    print(f"[{model_name}] Encoding passages...")
    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = passage_embeddings.shape[1]

    # Build FAISS inner product index (vectors normalized, equivalent to cosine similarity)
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
        top_idx = indices[0]
        results[qid] = [ids[i] for i in top_idx]

    return results


def save_raw_results_as_csv(
    results: dict,
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
):
    """
    Save {qid: [doc_id, ...]} as raw CSV:
    Columns: query, top_1, ..., top_k
    Rows aligned with df_queries order.
    """
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        retrieved_ids = results.get(qid, [])
        # If less than top_k, pad with empty strings on the right; if exceeds, truncate
        retrieved_ids = (retrieved_ids + [""] * top_k)[:top_k]
        rows.append([query] + retrieved_ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


def WeatherArchive_Retrieval_and_dump_raw():
    # Read queries and chunks, ensure id column exists
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)  # 需要 'query'
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv needs to contain 'query' column.")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)  # 需要 'Text'
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv needs to contain 'Text' column.")
    df_chunks = ensure_id_column(df_chunks, "id")

    # List of runnable models
    retrievers = [
        ("sbert", "sentence-transformers/msmarco-distilbert-base-tas-b"),
        ("splade", "naver/splade-cocondenser-ensembledistil"),
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
