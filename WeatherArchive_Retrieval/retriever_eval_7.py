# -*- coding: utf-8 -*-
# gemini-embedding-001
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import faiss
import tqdm

# --- Constants in your project ---
from constant.constants import (
    FILE_QUERY_ADDRESS,  # e.g., "data/queries.csv", needs column 'query'
    FILE_CONCATENATED_CHUNKS_ADDRESS,  # e.g., "data/concatenated_chunks.csv", needs column 'Text'
)
from WeatherArchive_Retrieval.utils import (
    BASE_ADDRESS,
)  # e.g., "WeatherArchive_Retrieval/retriever_eval"

# --- Google Gemini SDK ---
import google.generativeai as genai


# ============ Utility functions (consistent with original approach) ============


def ensure_id_column(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    Ensure DataFrame has unique id column; if not exists, generate using row numbers (string).
    """
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df


# ============ Global token bucket rate limiter (thread-safe) ============


class RateLimiter:
    """
    Simple token bucket: share RPM quota across threads, avoid triggering server-side rate limiting during concurrency.
    rpm: requests per minute allowed
    """

    def __init__(self, rpm: int):
        self.capacity = max(1, int(rpm))
        self.tokens = float(self.capacity)
        self.fill_rate = self.capacity / 60.0  # 每秒补充多少 token
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1):
        # Block until getting tokens
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.timestamp
                if elapsed > 0:
                    self.tokens = min(
                        self.capacity, self.tokens + elapsed * self.fill_rate
                    )
                    self.timestamp = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # Calculate wait time (wait without holding lock)
                needed = tokens - self.tokens
                wait_s = needed / self.fill_rate if self.fill_rate > 0 else 0.1
            time.sleep(wait_s)


# ============ Gemini embedding wrapper (supports multi-threading) ============


class GeminiEmbedder:
    """
    Unified wrapper for Google Gemini Embedding calls (multi-threading friendly).
    - Default 'models/gemini-embedding-001', fallback to 'models/text-embedding-004' if unavailable
    - embed_texts supports parallel when num_threads>1, each request goes through global RateLimiter
    """

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute: int = 300,  # 全局 RPM
        max_retries: int = 5,
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env} not detected. Please set your Google AI Studio API Key to this environment variable first."
            )
        genai.configure(api_key=api_key)

        self._candidates = list(model_candidates)
        self.model_name = None
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_minute)

        # Pre-select available model, ensure dimension consistency during concurrency
        self._ensure_working_model()

    # ---- Internal methods ----

    def _ensure_working_model(self):
        last_err = None
        for m in self._candidates:
            try:
                _ = self._embed_once(m, "healthcheck")
                self.model_name = m
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"没有可用的 Gemini 嵌入模型：{self._candidates}；最后错误：{last_err}"
        )

    def _embed_once(self, model_name: str, text: str):
        """
        Single call; returns list[float]
        """
        # Note: google-generativeai embeddings API parameter names vary slightly across versions.
        # Here we use embed_content(model=..., content=...)
        resp = genai.embed_content(model=model_name, content=text)
        # SDK usually returns {"embedding": [...]} or {"embedding": {"values": [...]}}, both compatible
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            emb = emb["values"]
        return emb

    def _embed_with_retry(self, text: str) -> np.ndarray:
        attempt = 0
        while True:
            # Global rate limiting
            self.rate_limiter.acquire(1)
            try:
                vec = self._embed_once(self.model_name, str(text))
                return np.asarray(vec, dtype="float32")
            except Exception as e:
                attempt += 1
                # If model 404 etc., try fallback once (rarely occurs, already health checked)
                if (
                    "model not found" in str(e).lower() or "404" in str(e).lower()
                ) and attempt == 1:
                    with self._lock:
                        # Scan candidates again to find an available one
                        self._ensure_working_model()
                    continue
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))  # 指数退避

    # ---- Public methods ----

    def embed_texts(self, texts, num_threads: int = 1, desc: str = "Encoding"):
        """
        Batch embedding, returns (N, D) float32 ndarray. Supports multi-threading concurrency.
        - num_threads=1: sequential execution (compatible with original behavior)
        - num_threads>1: use thread pool + global RateLimiter
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        N = len(texts)
        # Single-threaded path
        if num_threads <= 1:
            all_vecs = []
            for t in tqdm.tqdm(texts, desc=desc, total=N):
                all_vecs.append(self._embed_with_retry(t))
            emb = np.vstack(all_vecs)
        else:
            # Multi-threaded path: collect results by index, ensure return order matches input
            results = [None] * N
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                futures = {
                    ex.submit(self._embed_with_retry, texts[i]): i for i in range(N)
                }
                pbar = tqdm.tqdm(total=N, desc=f"{desc} (threads={num_threads})")
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        # Throw directly here, outer layer will catch; can also choose to write None and continue
                        raise RuntimeError(f"Embedding 失败（index={i}）：{e}")
                    finally:
                        pbar.update(1)
                pbar.close()
            emb = np.vstack(results)

        # Normalization (consistent with original implementation, use inner product index ~ cosine)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb


# ============ Same retrieval and storage interface as original implementation ============


def dense_retrieve_local_ids(
    df_queries: pd.DataFrame,
    df_chunks: pd.DataFrame,
    embedder: GeminiEmbedder,
    top_k: int = 100,
    num_threads: int = 15,  # New: control passages concurrency thread count
):
    """
    Use GeminiEmbedder + FAISS for dense retrieval, return top_k document **id list** for each query.
    - df_queries: needs columns ['id', 'query']
    - df_chunks: needs columns ['id', 'Text']
    Returns: {qid(str): [doc_id(str), ...]}
    """
    passages = df_chunks["Text"].astype(str).tolist()
    ids = df_chunks["id"].astype(str).tolist()
    if len(passages) == 0:
        raise ValueError(
            "Corpus is empty: no usable 'Text' found in concatenated_chunks.csv."
        )

    print("[Gemini] Encoding passages (multi-threading)...")
    passage_embeddings = embedder.embed_texts(
        passages,
        num_threads=num_threads,
        desc="Encoding passages",  # <<<<<< Concurrency!
    )
    dim = passage_embeddings.shape[1]

    # Build FAISS inner product index (vectors normalized -> inner product ≈ cosine)
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    results = {}
    print("[Gemini] Retrieving ...")
    # Queries are usually fewer, single-threaded is fine; if you have many queries too, can pass num_threads
    for _, row in tqdm.tqdm(
        df_queries.iterrows(),
        total=len(df_queries),
        desc="Retrieving with gemini-embedding-001",
    ):
        qid = str(row["id"])
        query = str(row["query"])
        q_emb = embedder.embed_texts([query], num_threads=1, desc="Encoding queries")
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

    # Initialize Gemini embedder (prioritize gemini-embedding-001, auto fallback to text-embedding-004 on failure)
    embedder = GeminiEmbedder(
        api_key_env="GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute=300,  # This is "global RPM", shared by 15 threads
        max_retries=5,
    )

    # Only one "retriever" entry, name follows original script style
    retrievers = [
        ("gemini-embedding-001", embedder),
    ]

    for short_name, emb in retrievers:
        print(f"\n=== Running retriever: {short_name} ===")
        try:
            results = dense_retrieve_local_ids(
                df_queries=df_queries,
                df_chunks=df_chunks,
                embedder=emb,
                top_k=100,
                num_threads=15,  # <<<<<< Set 15 threads
            )
            out_path = os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv")
            save_raw_results_as_csv(results, df_queries, out_path, top_k=100)
        except Exception as e:
            print(f"[Skipped] retriever '{short_name}' failed: {e}")


if __name__ == "__main__":
    WeatherArchive_Retrieval_and_dump_raw()
