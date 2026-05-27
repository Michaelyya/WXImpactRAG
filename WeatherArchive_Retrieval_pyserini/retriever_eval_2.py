# -*- coding: utf-8 -*-
"""
retriever_eval_2 (Pyserini) -- SBERT (dense) + SPLADE (learned sparse).

    * sbert  : sentence-transformers/msmarco-distilbert-base-tas-b
               -> Pyserini FAISS dense index (CLS pooling, L2-normalised),
                  searched with FaissSearcher + AutoQueryEncoder.
    * splade : naver/splade-cocondenser-ensembledistil
               -> Pyserini impact index (learned sparse), searched with
                  LuceneImpactSearcher.  SPLADE is genuinely a learned-sparse
                  model, so the Pyserini impact path is more faithful than the
                  original code's SentenceTransformer treatment.

Outputs: raw_model_result_sbert.csv, raw_model_result_splade.csv
"""

import os

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (
    BASE_ADDRESS,
    load_chunks,
    load_queries,
    make_auto_query_encoder,
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

    # --- SBERT (dense) ---
    try:
        model = "sentence-transformers/msmarco-distilbert-base-tas-b"
        qenc = make_auto_query_encoder(model, device=device, pooling="cls", l2_norm=True)
        res = run_dense_faiss(
            df_queries, df_chunks, "sbert", model, qenc,
            encoder_class="auto", pooling="cls", l2_norm=True, device=device, top_k=TOP_K,
        )
        save_raw_results_as_csv(
            res, df_queries, os.path.join(BASE_ADDRESS, "raw_model_result_sbert.csv"), TOP_K
        )
    except Exception as e:
        print(f"[Skipped] sbert failed: {e}")

    # --- SPLADE (learned sparse / impact) ---
    try:
        model = "naver/splade-cocondenser-ensembledistil"
        qenc = make_impact_query_encoder("splade", model, device=device)
        res = run_impact_lucene(
            df_queries, df_chunks, "splade", model, encoder_class="splade",
            query_encoder=qenc, weight_range=5, quant_range=256, device=device, top_k=TOP_K,
        )
        save_raw_results_as_csv(
            res, df_queries, os.path.join(BASE_ADDRESS, "raw_model_result_splade.csv"), TOP_K
        )
    except Exception as e:
        print(f"[Skipped] splade failed: {e}")


if __name__ == "__main__":
    main()
