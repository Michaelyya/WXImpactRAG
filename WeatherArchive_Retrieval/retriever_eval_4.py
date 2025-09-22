# Qwen
import os
import pandas as pd
import numpy as np
import tqdm
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# === Constants in your project (if not available, can directly replace with string paths) ===
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/queries.csv" (needs 'query' column)
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv" (needs 'Text' column)
)
from WeatherArchive_Retrieval.utils import (
    BASE_ADDRESS,
)  # e.g., "WeatherArchive_Retrieval/retriever_eval"


# -----------------------------
# Basic utilities
# -----------------------------
def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Ensure DF has unique string type id column; if not exists, generate using row numbers."""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


def save_raw_results_as_csv(
    results: dict,
    df_queries: pd.DataFrame,
    output_path: str,
    top_k: int = 100,
):
    """
    Write {qid: [doc_id, ...]} as raw CSV:
    Columns: query, top_1, ..., top_k
    Rows output in df_queries order.
    """
    rows = []
    for _, row in df_queries.iterrows():
        qid = str(row["id"])
        query = str(row["query"])
        ids = results.get(qid, [])
        ids = (ids + [""] * top_k)[:top_k]
        rows.append([query] + ids)

    columns = ["query"] + [f"top_{i}" for i in range(1, top_k + 1)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


# -----------------------------
# SentenceTransformer approach (single GPU/automatic)
# -----------------------------
def dense_retrieve_local_ids_st(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str,
    top_k: int = 100,
):
    """
    Use SentenceTransformer + FAISS; returns {qid: [doc_id, ...]}.
    """
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if not passages:
        raise ValueError(
            "Corpus is empty: concatenated_chunks.csv found no usable 'Text'."
        )

    print(f"[ST:{model_name}] Loading model...")
    model = SentenceTransformer(model_name)

    print(f"[ST:{model_name}] Encoding passages...")
    p_emb = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = p_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(p_emb)

    results = {}
    print(f"[ST:{model_name}] Retrieving...")
    for _, row in tqdm.tqdm(
        df_queries.iterrows(), total=len(df_queries), desc=f"Retrieving {model_name}"
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        k = min(top_k, len(passages))
        _, idx = index.search(q_emb, k)
        results[qid] = [ids[i] for i in idx[0]]

    return results


# -----------------------------
# Qwen Embedding approach (transformers, multi-GPU)
# -----------------------------
def encode_texts_qwen(texts, tokenizer, model, max_length=512, batch_size=16):
    """Apply CLS vector + L2 normalization to a batch of texts, return numpy array."""
    emb_chunks = []
    for i in tqdm.trange(0, len(texts), batch_size, desc="Encoding texts"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]  # [B, H]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)  # L2 norm
        emb_chunks.append(cls.cpu())
    return torch.cat(emb_chunks, dim=0).numpy()


def dense_retrieve_local_ids_qwen(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    model_name: str,
    top_k: int = 100,
):
    """
    Use Qwen Embedding (transformers + device_map='auto') for retrieval; returns {qid: [doc_id, ...]}.
    """
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if not passages:
        raise ValueError(
            "Corpus is empty: concatenated_chunks.csv found no usable 'Text'."
        )

    print(f"[Qwen:{model_name}] Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    print(f"[Qwen:{model_name}] Encoding passages...")
    p_emb = encode_texts_qwen(passages, tokenizer, model)
    dim = p_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(p_emb)

    results = {}
    print(f"[Qwen:{model_name}] Retrieving...")
    for _, row in tqdm.tqdm(
        df_queries.iterrows(), total=len(df_queries), desc=f"Retrieving {model_name}"
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = encode_texts_qwen([query], tokenizer, model)
        k = min(top_k, len(passages))
        _, idx = index.search(q_emb, k)
        results[qid] = [ids[i] for i in idx[0]]

    return results


# -----------------------------
# Specific model wrappers (add/remove as needed)
# -----------------------------
def retrieve_with_qwen3_4b(df_queries, df_chunks, top_k: int = 100):
    return dense_retrieve_local_ids_qwen(
        df_queries, df_chunks, "Qwen/Qwen3-Embedding-4B", top_k=top_k
    )


def retrieve_with_qwen3_8b(df_queries, df_chunks, top_k: int = 100):
    return dense_retrieve_local_ids_qwen(
        df_queries, df_chunks, "Qwen/Qwen3-Embedding-8B", top_k=top_k
    )


def retrieve_with_qwen3_0_6b(df_queries, df_chunks, top_k: int = 100):
    return dense_retrieve_local_ids_qwen(
        df_queries, df_chunks, "Qwen/Qwen3-Embedding-0.6B", top_k=top_k
    )


def retrieve_with_st_sbert(df_queries, df_chunks, top_k: int = 100):
    return dense_retrieve_local_ids_st(
        df_queries,
        df_chunks,
        "sentence-transformers/msmarco-distilbert-base-tas-b",
        top_k=top_k,
    )


# -----------------------------
# Main process: only output RAW CSV (query + top_1..top_100 ids)
# -----------------------------
def WeatherArchive_Retrieval_and_dump_raw():
    # Read data and ensure has id
    df_queries = pd.read_csv(FILE_QUERY_ADDRESS)
    if "query" not in df_queries.columns:
        raise ValueError("queries.csv needs to contain 'query' column.")
    df_queries = ensure_id_column(df_queries, "id")

    df_chunks = pd.read_csv(FILE_CONCATENATED_CHUNKS_ADDRESS)
    if "Text" not in df_chunks.columns:
        raise ValueError("concatenated_chunks.csv needs to contain 'Text' column.")
    df_chunks = ensure_id_column(df_chunks, "id")

    # You can add/remove models as needed
    retrievers = [
        ("qwen3-0_6b", retrieve_with_qwen3_0_6b)
        # If you want to compare with an ST model, you can also uncomment:
        # ("sbert",    retrieve_with_st_sbert),
    ]

    for name, fn in retrievers:
        print(f"\n=== Running retriever: {name} ===")
        try:
            results = fn(df_queries, df_chunks, top_k=100)
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{name}' failed: {e}")


if __name__ == "__main__":
    WeatherArchive_Retrieval_and_dump_raw()
