# -*- coding: utf-8 -*-
"""
retriever_eval_5 (Pyserini) -- OpenAI embedding models (dense).

Pyserini has no portable cross-version OpenAI document encoder, so we compute
the corpus embeddings with the OpenAI API (exactly as the original
retriever_eval_5), write them as a Pyserini embedding JSONL, build a Pyserini
FAISS index with ``pyserini.index.faiss``, and search it with FaissSearcher
driven by a small custom query encoder that calls the same API.  The retrieval
ranking is therefore owned by Pyserini's FaissSearcher.

Models: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002
Outputs: raw_model_result_openai-3-large.csv (and -small / -ada-002)

Requires OPENAI_API_KEY.
"""

import os

import dotenv
import numpy as np
import tqdm
from openai import OpenAI

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    run_dense_precomputed,
    save_raw_results_as_csv,
)

dotenv.load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TOP_K = 100


def encode_with_openai(texts, model="text-embedding-3-large", batch_size=64):
    """Embed a list of texts with the OpenAI Embeddings API (batched)."""
    embeddings = []
    for i in tqdm.trange(0, len(texts), batch_size, desc=f"Encoding with {model}"):
        batch_texts = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch_texts)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


class OpenAIQueryEncoder:
    """Custom Pyserini-compatible query encoder: encode(query) -> np.ndarray."""

    def __init__(self, model: str):
        self.model = model

    def encode(self, query: str):
        resp = client.embeddings.create(model=self.model, input=[query])
        return np.asarray(resp.data[0].embedding, dtype="float32")


def main():
    os.makedirs(BASE_ADDRESS, exist_ok=True)
    df_queries = load_queries()
    df_chunks = load_chunks()

    retrievers = [
        ("openai-3-large", "text-embedding-3-large"),
        ("openai-3-small", "text-embedding-3-small"),
        ("openai-ada-002", "text-embedding-ada-002"),
    ]

    passages = df_chunks["Text"].astype(str).tolist()

    for short_name, model in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model}) ===")
        try:
            doc_vecs = np.asarray(encode_with_openai(passages, model=model), dtype="float32")
            qenc = OpenAIQueryEncoder(model)
            res = run_dense_precomputed(
                df_queries, df_chunks, short_name, doc_vecs, qenc, top_k=TOP_K
            )
            save_raw_results_as_csv(
                res, df_queries,
                os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv"), TOP_K,
            )
        except Exception as e:
            print(f"[Skipped] {short_name} failed: {e}")


if __name__ == "__main__":
    main()
