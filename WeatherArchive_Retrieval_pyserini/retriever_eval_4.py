# -*- coding: utf-8 -*-
"""
retriever_eval_4 (Pyserini) -- Qwen3 embedding (dense).

    * qwen3-0_6b : Qwen/Qwen3-Embedding-0.6B
                   -> Pyserini FAISS dense index, searched with FaissSearcher +
                      AutoQueryEncoder.

The original retriever_eval_4 encoded with the transformers backend using the
CLS vector + L2 normalisation, so we mirror that here (pooling="cls",
l2_norm=True).  The 4B / 8B variants are left commented out, as in the original.

Output: raw_model_result_qwen3-0_6b.csv
"""

import os

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    make_auto_query_encoder,
    pick_device,
    run_dense_faiss,
    save_raw_results_as_csv,
)

TOP_K = 100


def main():
    os.makedirs(BASE_ADDRESS, exist_ok=True)
    df_queries = load_queries()
    df_chunks = load_chunks()
    device = pick_device()

    retrievers = [
        ("qwen3-0_6b", "Qwen/Qwen3-Embedding-0.6B"),
        # ("qwen3-4b", "Qwen/Qwen3-Embedding-4B"),
        # ("qwen3-8b", "Qwen/Qwen3-Embedding-8B"),
    ]

    for short_name, model in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model}) ===")
        try:
            qenc = make_auto_query_encoder(model, device=device, pooling="cls", l2_norm=True)
            res = run_dense_faiss(
                df_queries, df_chunks, short_name, model, qenc,
                encoder_class="auto", pooling="cls", l2_norm=True, device=device, top_k=TOP_K,
            )
            save_raw_results_as_csv(
                res, df_queries,
                os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv"), TOP_K,
            )
        except Exception as e:
            print(f"[Skipped] {short_name} failed: {e}")


if __name__ == "__main__":
    main()
