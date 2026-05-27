# -*- coding: utf-8 -*-
"""
retriever_eval_3 (Pyserini) -- ANCE (dense) + uniCOIL (learned sparse).

    * ance    : castorini/ance-msmarco-passage
                -> Pyserini FAISS dense index built with the native ANCE
                   encoder (--encoder-class ance), searched with FaissSearcher
                   + AnceQueryEncoder.
    * unicoil : castorini/unicoil-msmarco-passage
                -> Pyserini impact index (learned sparse), searched with
                   LuceneImpactSearcher.  uniCOIL is a learned-sparse model, so
                   the impact path is the correct Pyserini treatment.

Outputs: raw_model_result_ance.csv, raw_model_result_unicoil.csv
"""

import os

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    make_ance_query_encoder,
    make_impact_query_encoder,
    pick_device,
    run_dense_faiss,
    run_impact_lucene,
    save_raw_results_as_csv,
)

TOP_K = 100


def main():
    os.makedirs(BASE_ADDRESS, exist_ok=True)
    df_queries = load_queries()
    df_chunks = load_chunks()
    device = pick_device()

    # --- ANCE (dense) ---
    try:
        model = "castorini/ance-msmarco-passage"
        qenc = make_ance_query_encoder(model, device=device)
        res = run_dense_faiss(
            df_queries, df_chunks, "ance", model, qenc,
            encoder_class="ance", l2_norm=True, device=device, top_k=TOP_K,
        )
        save_raw_results_as_csv(
            res, df_queries, os.path.join(BASE_ADDRESS, "raw_model_result_ance.csv"), TOP_K
        )
    except Exception as e:
        print(f"[Skipped] ance failed: {e}")

    # --- uniCOIL (learned sparse / impact) ---
    try:
        model = "castorini/unicoil-msmarco-passage"
        qenc = make_impact_query_encoder("unicoil", model, device=device)
        res = run_impact_lucene(
            df_queries, df_chunks, "unicoil", model, encoder_class="unicoil",
            query_encoder=qenc, device=device, top_k=TOP_K,
        )
        save_raw_results_as_csv(
            res, df_queries, os.path.join(BASE_ADDRESS, "raw_model_result_unicoil.csv"), TOP_K
        )
    except Exception as e:
        print(f"[Skipped] unicoil failed: {e}")


if __name__ == "__main__":
    main()
