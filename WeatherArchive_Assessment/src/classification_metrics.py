import pandas as pd
import numpy as np
import re
import os
import glob
import argparse
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def clean_label(text):
    """Extract the first alphabetical word from a label and normalize it (e.g., 'health```' -> 'health')."""
    if pd.isna(text):
        return "NA"
    s = str(text).strip()
    if s.lower() in ["nan", "none", "[na]", "na", ""]:
        return "NA"
    # Remove common surrounding punctuation/backticks/quotes
    s = s.replace("`", "").replace('"', "").replace("'", "")
    # Take the first alphabetical token
    match = re.search(r"[A-Za-z]+", s)
    return match.group(0).lower() if match else "NA"


def extract_first_number(text):
    if pd.isna(text) or text == "":
        return "NA"

    text_str = str(text).strip()
    if text_str.lower() in ["nan", "none", "[na]", "na"]:
        return "NA"

    match = re.search(r"\d+", text_str)
    return int(match.group()) if match else None


def evaluate_mcq_model(ground_truth_file, model_file):
    gt_df = pd.read_csv(ground_truth_file)
    model_df = pd.read_csv(model_file)

    results = {}
    mcq_columns = [
        "exposure",
        "sensitivity",
        "adaptability",
        "temporal",
        "functional",
        "spatial",
    ]

    for col in mcq_columns:
        if col not in gt_df.columns or col not in model_df.columns:
            continue

        gt_values = gt_df[col].apply(clean_label)
        model_values = model_df[col].apply(clean_label)

        results[f"{col}_accuracy"] = accuracy_score(gt_values, model_values)
        results[f"{col}_f1"] = f1_score(
            gt_values, model_values, average="macro", zero_division=0
        )
        results[f"{col}_precision"] = precision_score(
            gt_values, model_values, average="macro", zero_division=0
        )
        results[f"{col}_recall"] = recall_score(
            gt_values, model_values, average="macro", zero_division=0
        )

    # Calculate overall metrics
    all_accuracy_scores = [
        results[f"{col}_accuracy"]
        for col in mcq_columns
        if f"{col}_accuracy" in results and not pd.isna(results[f"{col}_accuracy"])
    ]
    all_f1_scores = [
        results[f"{col}_f1"]
        for col in mcq_columns
        if f"{col}_f1" in results and not pd.isna(results[f"{col}_f1"])
    ]
    all_recall_scores = [
        results[f"{col}_recall"]
        for col in mcq_columns
        if f"{col}_recall" in results and not pd.isna(results[f"{col}_recall"])
    ]

    if all_accuracy_scores:
        results["overall_accuracy"] = np.mean(all_accuracy_scores)
    if all_f1_scores:
        results["overall_f1"] = np.mean(all_f1_scores)
    if all_recall_scores:
        results["overall_recall"] = np.mean(all_recall_scores)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MCQ model outputs against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        "-gt",
        default="data/ground_truth_climate.csv",
        help="Path to ground truth CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="WeatherArchive_Assessment/output",
        help="Directory containing model output CSV files",
    )
    parser.add_argument(
        "--output-file",
        "-f",
        default="classification_evaluation_summary.csv",
        help="Output CSV file for summary results",
    )

    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.output_dir, "*.csv"))
    model_files = [
        f
        for f in all_files
        if not os.path.basename(f).startswith("rag_generation")
        and not os.path.basename(f).startswith("ground_truth")
    ]

    all_results = []
    for model_file in sorted(model_files):
        model_name = os.path.basename(model_file).replace("-results.csv", "")
        results = evaluate_mcq_model(args.ground_truth, model_file)
        if results:
            results["model_name"] = model_name
            all_results.append(results)
            print(f"  ✓ Completed")
        else:
            print(f"  ✗ Failed - No results")

    summary_df = pd.DataFrame(all_results)
    cols = ["model_name"] + [col for col in summary_df.columns if col != "model_name"]
    summary_df = summary_df[cols]
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

    summary_df.to_csv(args.output_file, index=False)
    print(f"MCQ Summary saved to: {args.output_file}")


if __name__ == "__main__":
    main()
