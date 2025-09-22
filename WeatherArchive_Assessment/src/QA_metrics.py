import pandas as pd
import numpy as np
import re
import os
import glob
import argparse
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import bert_score
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # breakpoint()
    return f1


def acc_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if ground_truth in prediction:
        return 1
    return 0


def llm_judge_answer(oracle_answer, generated_answer, model_name="gpt-4.1"):
    prompt = f"""You are an expert evaluator. Compare the oracle answer with the generated answer and determine if the generated answer COVERS the key information stated in the oracle answer.

Oracle Answer: {oracle_answer}

Generated Answer: {generated_answer}

Task: Determine if the generated answer COVERS the key information from the oracle answer.

Consider:
- Does the generated answer contain the main points from the oracle answer?
- Is the information accurate and relevant?
- Does it address the same question/topic?

Output ONLY: "true" if it covers, "false" if it doesn't cover."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
    )

    judge_output = response.choices[0].message.content.strip().lower()

    if "true" in judge_output:
        return 1
    else:
        return 0


def evaluate_qa_model(ground_truth_file, model_file):
    gt_df = pd.read_csv(ground_truth_file)
    model_df = pd.read_csv(model_file)

    results = {}

    gt_answers = gt_df["rag_answer"].astype(str)
    model_answers = model_df["rag_answer"].astype(str)

    # BLEU Score
    bleu_scores = []
    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            bleu_scores.append(1.0 if gt_answer == model_answer else 0.0)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            bleu_scores.append(0.0)
        else:
            gt_words = gt_answer.split()
            model_words = model_answer.split()
            bleu = sentence_bleu([gt_words], model_words) if model_words else 0.0
            bleu_scores.append(bleu)

    results["bleu_score"] = np.mean(bleu_scores)

    # ROUGE Scores
    rouge = Rouge()
    rouge_1_scores = []
    rouge_l_scores = []

    for gt_answer, model_answer in zip(gt_answers, model_answers):
        if gt_answer == "nan" or model_answer == "nan":
            score = 1.0 if gt_answer == model_answer else 0.0
            rouge_1_scores.append(score)
            rouge_l_scores.append(score)
        elif gt_answer.strip() == "" or model_answer.strip() == "":
            rouge_1_scores.append(0.0)
            rouge_l_scores.append(0.0)
        else:
            try:
                scores = rouge.get_scores(model_answer, gt_answer)
                rouge_1_scores.append(scores[0]["rouge-1"]["f"])
                rouge_l_scores.append(scores[0]["rouge-l"]["f"])
            except:
                rouge_1_scores.append(0.0)
                rouge_l_scores.append(0.0)

    results["rouge_1"] = np.mean(rouge_1_scores)
    results["rouge_l"] = np.mean(rouge_l_scores)

    # BERTScore
    try:
        P, R, F1 = bert_score.score(
            model_answers.tolist(), gt_answers.tolist(), lang="en", verbose=False
        )
        results["bertscore_f1"] = F1.mean().item()
    except Exception as e:
        print(f"Warning: BERTScore calculation failed: {e}")
        results["bertscore_f1"] = float("nan")

    # f1_scores = []
    # for gt_answer, model_answer in zip(gt_answers, model_answers):
    #     if gt_answer == "nan" or model_answer == "nan":
    #         f1_scores.append(1.0 if gt_answer == model_answer else 0.0)
    #     elif gt_answer.strip() == "" or model_answer.strip() == "":
    #         f1_scores.append(0.0)
    #     else:
    #         f1 = f1_score(model_answer, gt_answer)
    #         f1_scores.append(f1)

    # results["f1"] = np.mean(f1_scores)

    # accuracy_scores = []
    # for gt_answer, model_answer in zip(gt_answers, model_answers):
    #     if gt_answer == "nan" or model_answer == "nan":
    #         accuracy_scores.append(1.0 if gt_answer == model_answer else 0.0)
    #     elif gt_answer.strip() == "" or model_answer.strip() == "":
    #         accuracy_scores.append(0.0)
    #     else:
    #         acc = acc_score(model_answer, gt_answer)
    #         accuracy_scores.append(acc)

    # results["accuracy"] = np.mean(accuracy_scores)

    # # LLM-as-a-Judge evaluation
    # print("  Running LLM-as-a-Judge evaluation...")
    # llm_judge_scores = []
    # for i, (gt_answer, model_answer) in enumerate(zip(gt_answers, model_answers)):
    #     if gt_answer == "nan" or model_answer == "nan":
    #         llm_judge_scores.append(1.0 if gt_answer == model_answer else 0.0)
    #     elif gt_answer.strip() == "" or model_answer.strip() == "":
    #         llm_judge_scores.append(0.0)
    #     else:
    #         try:
    #             judge_score = llm_judge_answer(gt_answer, model_answer)
    #             llm_judge_scores.append(judge_score)
    #         except Exception as e:
    #             print(f"Warning: LLM judge failed for sample {i}: {e}")
    #             llm_judge_scores.append(0.0)

    # results["llm_judge_score"] = np.mean(llm_judge_scores)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA model outputs against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        "-gt",
        default="WeatherArchive_Assessment/output/ground_truth_QA.csv",
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
        default="QA_evaluation_summary.csv",
        help="Output CSV file for summary results",
    )

    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.output_dir, "rag_generation_*.csv"))
    model_files = [
        f for f in all_files if not os.path.basename(f).endswith("gpt_4.1.csv")
    ]

    print("=" * 80)
    print("EVALUATING QA MODELS AGAINST GROUND TRUTH")
    print("=" * 80)
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Found {len(model_files)} QA model result files")
    print("=" * 80)

    all_results = []
    for model_file in sorted(model_files):
        model_name = (
            os.path.basename(model_file)
            .replace("rag_generation_", "")
            .replace(".csv", "")
        )
        print(f"Evaluating: {model_name}")

        try:
            results = evaluate_qa_model(args.ground_truth, model_file)
            if results:
                results["model_name"] = model_name
                all_results.append(results)
                print(f"  ✓ Completed")
            else:
                print(f"  ✗ Failed - No results")
        except Exception as e:
            print(f"  ✗ Failed - Error: {e}")

    if all_results:
        print(f"\n{'='*80}")
        print("CREATING QA SUMMARY TABLE")
        print("=" * 80)

        summary_df = pd.DataFrame(all_results)
        cols = ["model_name"] + [
            col for col in summary_df.columns if col != "model_name"
        ]
        summary_df = summary_df[cols]

        # Round numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

        summary_df.to_csv(args.output_file, index=False)
        print(f"QA Summary saved to: {args.output_file}")

        print("\nQA SUMMARY TABLE:")
        print("=" * 120)
        print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("QA MODEL EVALUATIONS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
