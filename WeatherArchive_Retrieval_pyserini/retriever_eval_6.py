# -*- coding: utf-8 -*-
"""
retriever_eval_6 (Pyserini) -- Arctic + Granite embeddings (dense).

    * arctic  : Snowflake/snowflake-arctic-embed-l-v2.0  (CLS pooling)
    * granite : ibm-granite/granite-embedding-english-r2 (mean pooling)

Both go through a Pyserini FAISS dense index searched with FaissSearcher +
AutoQueryEncoder.  Pooling matches each model's published configuration; the
document and query sides use the same pooling so the index stays consistent.

Outputs: raw_model_result_arctic.csv, raw_model_result_granite.csv
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

    # (short_name, model, pooling)
    retrievers = [
        ("arctic", "Snowflake/snowflake-arctic-embed-l-v2.0", "cls"),
        ("granite", "ibm-granite/granite-embedding-english-r2", "mean"),
    ]

    for short_name, model, pooling in retrievers:
        print(f"\n=== Running retriever: {short_name} ({model}, pooling={pooling}) ===")
        try:
            qenc = make_auto_query_encoder(model, device=device, pooling=pooling, l2_norm=True)
            res = run_dense_faiss(
                df_queries, df_chunks, short_name, model, qenc,
                encoder_class="auto", pooling=pooling, l2_norm=True, device=device, top_k=TOP_K,
            )
            save_raw_results_as_csv(
                res, df_queries,
                os.path.join(BASE_ADDRESS, f"raw_model_result_{short_name}.csv"), TOP_K,
            )
        except Exception as e:
            print(f"[Skipped] {short_name} failed: {e}")


if __name__ == "__main__":
    main()
