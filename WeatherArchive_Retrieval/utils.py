BASE_ADDRESS = "WeatherArchive_Retrieval/retriever_eval"

import pandas as pd
from typing import Dict, List

# New: gold now comes from these two files
from constant.constants import FILE_QUERY_ADDRESS, FILE_CORRECT_PASSAGES_ADDRESS


def evaluate_retriever_performance(
    retrieval_results: Dict[str, List[str]],
    output_path: str,
    ks: List[int] = (1, 5, 10, 50, 100),
):
    """
    Evaluate retrieval performance:
    - retrieval_results: {qid(str): [passage_text, ...]}
    - gold: now comes from 'correct_passage' in correct_passages.csv, one-to-one correspondence with query.csv
    - calculate recall@k and mrr@k, k defaults to include 1/5/10/50/100
    """
    # Read query and gold, align ids
    qdf = pd.read_csv(FILE_QUERY_ADDRESS)
    if "id" not in qdf.columns:
        qdf = qdf.reset_index().rename(columns={"index": "id"})
    qdf["id"] = qdf["id"].astype(str)

    gdf = pd.read_csv(FILE_CORRECT_PASSAGES_ADDRESS)
    if "correct_passage" not in gdf.columns or len(gdf) != len(qdf):
        raise ValueError(
            "correct_passages.csv must contain 'correct_passage' column, and row count must align with query.csv."
        )

    # Attach gold to qdf, ensure one-to-one correspondence (row alignment)
    qdf = qdf.copy()
    qdf["correct_passage"] = gdf["correct_passage"].astype(str)

    results = []
    ks = list(sorted(set(int(k) for k in ks)))  # Remove duplicates and sort, robustness

    for _, row in qdf.iterrows():
        qid = row["id"]
        query = str(row["query"])
        golden_answer = str(row["correct_passage"])

        retrieved_passages = retrieval_results.get(qid, [])

        # Initialize metrics for each k
        hit_k = {f"recall@{k}": 0 for k in ks}
        mrr_k = {f"mrr@{k}": 0.0 for k in ks}
        hit_rank = -1  # First hit rank (1-based), -1 if no hit

        # Iterate through retrieval list, find first hit position
        max_k = max(ks) if ks else len(retrieved_passages)
        for rank, p in enumerate(retrieved_passages[:max_k], start=1):
            if str(p).strip() == golden_answer.strip():
                hit_rank = rank
                for k in ks:
                    if rank <= k:
                        hit_k[f"recall@{k}"] = 1
                        mrr_k[f"mrr@{k}"] = 1.0 / rank
                break  # Only count first hit

        results.append(
            {
                "id": qid,
                "query": query,
                "hit_rank": hit_rank,
                **hit_k,
                **mrr_k,
            }
        )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
