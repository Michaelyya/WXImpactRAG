# OpenAI embedding models
import os
import numpy as np
import pandas as pd
import faiss
import tqdm
from openai import OpenAI
from WeatherArchive_Retrieval.utils import BASE_ADDRESS
from constant.constants import (
    FILE_QUERY_ADDRESS,  # needs to contain 'query' column
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # needs to contain 'Text' column; if no 'id' will auto-generate
)

import dotenv

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------- Utils ----------
def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Ensure unique string type id column exists; if not, generate using row numbers."""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def encode_with_openai(texts, model="text-embedding-3-large", batch_size=64):
    """
    Use OpenAI Embeddings API to encode texts (batch).
    Returns: List[List[float]]
    """
    embeddings = []
    for i in tqdm.trange(0, len(texts), batch_size, desc=f"Encoding with {model}"):
        batch_texts = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch_texts)
        batch_embeddings = [d.embedding for d in resp.data]
        embeddings.extend(batch_embeddings)
    return embeddings


def dense_retrieve_openai_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str = "text-embedding-3-large",
    top_k: int = 100,
):
    """
    Local dense retrieval based on OpenAI Embeddings + FAISS.
    Returns: {qid(str): [doc_id(str), ...]} only contains top_k document ids.
    """
    # Corpus and id (order will determine vector and index alignment)
    passages = df_chunks["Text"].astype(str).tolist()
    doc_ids = df_chunks["id"].astype(str).tolist()
    if not passages:
        raise ValueError("concatenated_chunks.csv 'Text' is empty.")

    # 1) Encode corpus and build index (inner product + normalization = cosine similarity)
    passage_embs = encode_with_openai(passages, model=model_name)
    passage_embs = np.asarray(passage_embs, dtype="float32")
    dim = passage_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs)

    # 2) Encode query one by one (or batch) and retrieve
    results = {}
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc=f"Retrieving with {model_name}",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = encode_with_openai([query], model=model_name)[0]
        q_emb = np.asarray([q_emb], dtype="float32")
        k = min(top_k, len(passages))
        scores, indices = index.search(q_emb, k)
        top_idx = indices[0]
        results[qid] = [doc_ids[i] for i in top_idx]
    return results


def save_raw_results_as_csv(
    results: dict, df_queries: pd.DataFrame, output_path: str, top_k: int = 100
):
    """Save {qid: [doc_id,...]} as CSV: query, top_1..top_k (aligned with df_queries original order)."""
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        ids = results.get(qid, [])
        ids = (ids + [""] * top_k)[
            :top_k
        ]  # Pad with empty if insufficient, truncate if exceeds
        rows.append([query] + ids)

    cols = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


# ---------- Main ----------
def WeatherArchive_Retrieval_and_dump_raw():
    # Read data
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv needs to contain 'query' column.")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv needs to contain 'Text' column.")
    df_chunks = ensure_id_column(df_chunks, "id")

    # OpenAI embedding models to run (can add/remove as needed)
    retrievers = [
        ("openai-3-large", "text-embedding-3-large"),
        ("openai-3-small", "text-embedding-3-small"),
        ("openai-ada-002", "text-embedding-ada-002"),
    ]

    for short_name, model_name in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model_name}) ===")
        try:
            results = dense_retrieve_openai_ids(
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
