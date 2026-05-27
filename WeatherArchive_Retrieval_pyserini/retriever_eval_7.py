# -*- coding: utf-8 -*-
"""
retriever_eval_7 (Pyserini) -- Google Gemini embedding (dense).

Pyserini ships no Gemini encoder, so (as with the OpenAI script) we embed the
corpus with the Gemini API -- reusing the original multi-threaded, rate-limited
embedder -- write the (L2-normalised) vectors as a Pyserini embedding JSONL,
build a Pyserini FAISS index, and search it with FaissSearcher + a custom
Gemini query encoder.  Ranking is owned by Pyserini's FaissSearcher.

Model: models/gemini-embedding-001 (fallback models/text-embedding-004)
Output: raw_model_result_gemini-embedding-001.csv

Requires GOOGLE_API_KEY.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tqdm
import google.generativeai as genai

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    run_dense_precomputed,
    save_raw_results_as_csv,
)

TOP_K = 100


class RateLimiter:
    """Thread-safe token bucket sharing one RPM quota across threads."""

    def __init__(self, rpm: int):
        self.capacity = max(1, int(rpm))
        self.tokens = float(self.capacity)
        self.fill_rate = self.capacity / 60.0
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.timestamp
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                    self.timestamp = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
                wait_s = needed / self.fill_rate if self.fill_rate > 0 else 0.1
            time.sleep(wait_s)


class GeminiEmbedder:
    """Multi-threaded, rate-limited Gemini embedding wrapper (L2-normalised)."""

    def __init__(
        self,
        api_key_env: str = "GOOGLE_API_KEY",
        model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
        requests_per_minute: int = 300,
        max_retries: int = 5,
    ):
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env} not set. Provide a Google AI "
                f"Studio API key."
            )
        genai.configure(api_key=api_key)

        self._candidates = list(model_candidates)
        self.model_name = None
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_minute)
        self._ensure_working_model()

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
            f"No usable Gemini embedding model in {self._candidates}; last error: {last_err}"
        )

    def _embed_once(self, model_name: str, text: str):
        resp = genai.embed_content(model=model_name, content=text)
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            emb = emb["values"]
        return emb

    def _embed_with_retry(self, text: str) -> np.ndarray:
        attempt = 0
        while True:
            self.rate_limiter.acquire(1)
            try:
                vec = self._embed_once(self.model_name, str(text))
                return np.asarray(vec, dtype="float32")
            except Exception as e:
                attempt += 1
                if (
                    "model not found" in str(e).lower() or "404" in str(e).lower()
                ) and attempt == 1:
                    with self._lock:
                        self._ensure_working_model()
                    continue
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

    def embed_texts(self, texts, num_threads: int = 1, desc: str = "Encoding") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        N = len(texts)
        if num_threads <= 1:
            all_vecs = [self._embed_with_retry(t) for t in tqdm.tqdm(texts, desc=desc, total=N)]
            emb = np.vstack(all_vecs)
        else:
            results = [None] * N
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                futures = {ex.submit(self._embed_with_retry, texts[i]): i for i in range(N)}
                pbar = tqdm.tqdm(total=N, desc=f"{desc} (threads={num_threads})")
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Embedding failed (index={i}): {e}")
                    finally:
                        pbar.update(1)
                pbar.close()
            emb = np.vstack(results)

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb / norms


class GeminiQueryEncoder:
    """Custom Pyserini-compatible query encoder backed by GeminiEmbedder."""

    def __init__(self, embedder: GeminiEmbedder):
        self.embedder = embedder

    def encode(self, query: str) -> np.ndarray:
        return self.embedder.embed_texts([query], num_threads=1)[0].astype("float32")


def main():
    os.makedirs(BASE_ADDRESS, exist_ok=True)
    df_queries = load_queries()
    df_chunks = load_chunks()

    try:
        embedder = GeminiEmbedder(
            api_key_env="GOOGLE_API_KEY",
            model_candidates=("models/gemini-embedding-001", "models/text-embedding-004"),
            requests_per_minute=300,
            max_retries=5,
        )
        passages = df_chunks["Text"].astype(str).tolist()
        doc_vecs = embedder.embed_texts(passages, num_threads=15, desc="Encoding passages")
        qenc = GeminiQueryEncoder(embedder)
        res = run_dense_precomputed(
            df_queries, df_chunks, "gemini-embedding-001", doc_vecs, qenc, top_k=TOP_K
        )
        save_raw_results_as_csv(
            res, df_queries,
            os.path.join(BASE_ADDRESS, "raw_model_result_gemini-embedding-001.csv"), TOP_K,
        )
    except Exception as e:
        print(f"[Skipped] gemini-embedding-001 failed: {e}")


if __name__ == "__main__":
    main()
