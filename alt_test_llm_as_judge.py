# alt_test_llm_as_judge.py
# Requirements: pandas, numpy, scipy (for t-test)
# Usage:
#   python alt_test_llm_as_judge.py \
#     --humans HumanAnnotatedAllen.csv HumanAnnotatedAngela.csv HumanAnnotatedDanny.csv HumanAnnotatedMichael.csv \
#     --llm ground_truth_climate.csv \
#     --epsilon 0.15 \
#     --alpha 0.05

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from collections import defaultdict

FIELDS = [
    "exposure",
    "sensitivity",
    "adaptability",
    "temporal",
    "functional",
    "spatial",
]


def read_human_file(path: str, annotator_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 标准化列名
    df.columns = [c.strip() for c in df.columns]
    # 统一关键列名到小写，以便合并
    rename_map = {
        "Query": "query",
        "Context": "context",
        "RAG-Answer": "rag_answer",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    # 只保留关心列
    keep_cols = ["query", "context"] + FIELDS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    out = df[keep_cols].copy()
    out["annotator"] = annotator_name
    return out


def read_llm_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # 统一列名
    rename_map = {
        "query": "query",
        "correct_passage_context": "context",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    keep_cols = ["query", "context"] + FIELDS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    return df[keep_cols].copy()


def by_fdr(pvals, q=0.05):
    """
    Benjamini–Yekutieli (BY) 控错：适合任意相关性，较保守。
    返回：reject 布尔数组（是否显著），以及调整后的阈值信息。
    """
    m = len(pvals)
    p_order = np.argsort(pvals)
    p_sorted = np.array(pvals)[p_order]

    # 调整因子 H_m = sum_{i=1..m} 1/i
    H_m = np.sum(1.0 / np.arange(1, m + 1))
    crit = (np.arange(1, m + 1) / m) * (q / H_m)  # BY 临界线

    # 找到最大的 k 使 p_(k) <= crit_k
    k = np.where(p_sorted <= crit)[0]
    reject = np.zeros(m, dtype=bool)
    if len(k) > 0:
        kmax = k.max()
        reject[p_order[: kmax + 1]] = True
    return reject, crit


def compute_S_for_candidate(
    sample_group: pd.DataFrame, candidate_row: pd.Series, exclude_annotator: str
):
    """
    对单个样本，计算候选者（LLM 或单个人类）的 S 分数：
    对 6 个字段分别计算“与其余人类相同的比例”，再取平均。
    sample_group: 该 (query, context) 下所有人类的行（含被排除对象）
    candidate_row: 候选者（Series），含六个字段取值
    exclude_annotator: 留一出的“被排除人类”名字（若候选正是该人，则不计入“其余人类”）
    """
    others = sample_group[sample_group["annotator"] != exclude_annotator]
    # 其余人类至少 1 人才有意义
    if len(others) == 0:
        return np.nan
    per_field_scores = []
    for f in FIELDS:
        # 其余人类在该字段上的众包标签列表
        vals = others[f].dropna().astype(str).tolist()
        if len(vals) == 0:
            per_field_scores.append(np.nan)
            continue
        candidate_val = str(candidate_row[f])
        agree = sum(1 for v in vals if v == candidate_val)
        per_field_scores.append(agree / len(vals))
    # 去掉 NaN 后平均
    arr = np.array(per_field_scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(arr.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--humans", nargs="+", required=True, help="四个（或更多）人类标注文件路径"
    )
    parser.add_argument(
        "--llm", required=True, help="LLM 结果 CSV 路径（ground_truth_climate.csv）"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.15, help="epsilon 成本折扣，默认 0.15"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="显著性水平（单侧检验），默认 0.05"
    )
    args = parser.parse_args()

    # 读入人类
    human_dfs = []
    for p in args.humans:
        name = os.path.splitext(os.path.basename(p))[0]
        human_dfs.append(read_human_file(p, name))
    humans = pd.concat(human_dfs, ignore_index=True)

    # 读入 LLM
    llm = read_llm_file(args.llm)
    llm["annotator"] = "LLM"

    # 我们只在 人类与 LLM 都存在 的 (query, context) 上比较
    # 先找出交集
    key_cols = ["query", "context"]
    human_pairs = humans[key_cols].drop_duplicates()
    llm_pairs = llm[key_cols].drop_duplicates()
    common_pairs = human_pairs.merge(llm_pairs, on=key_cols, how="inner")

    if len(common_pairs) == 0:
        raise RuntimeError("没有 (query, context) 同时被 LLM 与人类标注的样本。")

    # 只保留交集
    humans = humans.merge(common_pairs, on=key_cols, how="inner")
    llm = llm.merge(common_pairs, on=key_cols, how="inner")

    annotators = sorted(humans["annotator"].unique().tolist())
    print(f"发现人类标注者: {annotators}")
    print(f"可比较样本数（交集）：{len(common_pairs)}")
    print(f"使用 epsilon={args.epsilon}, alpha={args.alpha}\n")

    # 为加速，把 LLM 按 (query, context) 索引起来
    llm_idx = llm.set_index(key_cols)

    results = []
    pvals = []

    # 针对每位人类做“留一出”
    for hj in annotators:
        # 该人类的所有样本
        hj_rows = humans[humans["annotator"] == hj].copy()
        # 只保留 LLM 也有的样本（之前已经是交集，但防御性写法）
        hj_rows = hj_rows.merge(
            llm_idx.reset_index()[key_cols], on=key_cols, how="inner"
        )
        if len(hj_rows) == 0:
            print(f"[跳过] {hj}: 无可比较样本。")
            results.append(
                {
                    "annotator": hj,
                    "n": 0,
                    "mean_diff": np.nan,
                    "t_stat": np.nan,
                    "p_value": 1.0,
                    "reject": False,
                }
            )
            pvals.append(1.0)
            continue

        diffs = []
        # 按样本分组（为了能拿到“其余人类”的同一组）
        grouped = humans.groupby(key_cols, sort=False)

        for _, row in hj_rows.iterrows():
            key = (row["query"], row["context"])
            # 该样本下的所有人类（含被排除者）
            sample_group = grouped.get_group(key)

            # 候选 = 该人类
            S_hj = compute_S_for_candidate(sample_group, row, exclude_annotator=hj)

            # 候选 = LLM（从 LLM 表里取这条样本的六个标签）
            llm_row = llm_idx.loc[key]
            S_llm = compute_S_for_candidate(sample_group, llm_row, exclude_annotator=hj)

            if np.isnan(S_hj) or np.isnan(S_llm):
                continue

            diffs.append(S_llm - S_hj)  # 每条样本的差值

        diffs = np.array(diffs, dtype=float)
        n = len(diffs)
        if n == 0:
            print(f"[跳过] {hj}: 有样本但无法计算有效分数（可能其余人类为空）。")
            results.append(
                {
                    "annotator": hj,
                    "n": 0,
                    "mean_diff": np.nan,
                    "t_stat": np.nan,
                    "p_value": 1.0,
                    "reject": False,
                }
            )
            pvals.append(1.0)
            continue

        # 单样本 t 检验：H0: mean(diffs) <= -epsilon  vs  H1: mean(diffs) > -epsilon
        # 等价于对 diffs + epsilon 做 H0: mean <= 0 vs H1: mean > 0
        diffs_shift = diffs + args.epsilon
        t_res = ttest_1samp(diffs_shift, popmean=0.0, alternative="greater")
        pval = float(t_res.pvalue)
        pvals.append(pval)

        results.append(
            {
                "annotator": hj,
                "n": n,
                "mean_diff": float(
                    np.mean(diffs)
                ),  # 未加 epsilon 的平均差 (LLM - human)
                "t_stat": float(t_res.statistic),
                "p_value": pval,
                "reject": None,  # 先占位，FDR 后回填
            }
        )

    # FDR（BY）校正与显著性判断
    rej, _ = by_fdr(pvals, q=args.alpha)
    for i, r in enumerate(results):
        r["reject"] = bool(rej[i])

    # 计算 ω
    valid = [r for r in results if not np.isnan(r["p_value"])]
    if len(valid) == 0:
        raise RuntimeError("无有效检验结果。")
    omega = np.mean([r["reject"] for r in results])

    # 汇报
    df_out = pd.DataFrame(results)
    print("=== 每位人类的检验结果（留一出）===")
    print(df_out.to_string(index=False))
    print("\n=== 总结 ===")
    print(f"Winning Rate ω = {omega:.2f}")
    if omega >= 0.5:
        print("结论：在本研究设置下（给定 ε 与 α），LLM 可以替代人类标注者。")
    else:
        print("结论：在本研究设置下（给定 ε 与 α），LLM 尚不足以替代人类标注者。")


if __name__ == "__main__":
    main()
