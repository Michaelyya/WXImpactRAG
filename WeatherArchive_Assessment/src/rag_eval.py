import os
import dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
from typing import Dict, List
from huggingface_hub import login
import torch.cuda

# Set HuggingFace cache to scratch directory to avoid disk quota issues
os.environ["HF_HOME"] = "/scratch/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/hf_cache"

from constant.constants import (
    HF_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
)
from constant.climate_framework import RAG_Answering_prompt

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Hugging Face authentication
API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
if API_KEY:
    login(token=API_KEY)
    print("Hugging Face authentication successful")


def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        print(f"Available: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB used")
    else:
        print("No GPU available")


def initialize_hf_model(model_name: str):
    print(f"Loading Hugging Face model: {model_name}")

    check_gpu_memory()

    # Use scratch directory for model caching to avoid disk quota issues
    cache_dir = "/scratch/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=config,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    model.gradient_checkpointing_enable()

    return model, tokenizer


def generate_hf_answer(prompt: str, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result.startswith(prompt):
        result = result[len(prompt) :].strip()

    return result


def generate_openai_answer(prompt: str, model_name: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def load_models(selected_models: List[str]):
    loaded_models = {}

    for model_id in selected_models:
        if model_id in HF_MODELS:
            model_name = HF_MODELS[model_id]
            model, tokenizer = initialize_hf_model(model_name)
            loaded_models[model_id] = {
                "type": "hf",
                "name": model_name,
                "model": model,
                "tokenizer": tokenizer,
            }
            print(f"Successfully loaded HF model: {model_name}")

        elif model_id in OPENAI_MODELS:
            model_name = OPENAI_MODELS[model_id]
            loaded_models[model_id] = {"type": "openai", "name": model_name}
            print(f"Successfully loaded OpenAI model: {model_name}")
        elif model_id in DEEPSEEK_MODELS:
            model_name = DEEPSEEK_MODELS[model_id]
            loaded_models[model_id] = {"type": "closed-model", "name": model_name}
            print(f"Successfully loaded DeepSeeK model: {model_name}")

    return loaded_models


def generate_rag_answer(
    query: str, top3_contexts: List[str], model_id: str, loaded_models: Dict
):
    combined_context = "\n\n".join(
        [f"Context {i+1}: {context}" for i, context in enumerate(top3_contexts)]
    )

    prompt = RAG_Answering_prompt.format(query=query, context=combined_context)

    model_info = loaded_models[model_id]

    if model_info["type"] == "hf":
        response = generate_hf_answer(
            prompt, model_info["model"], model_info["tokenizer"]
        )
    elif model_info["type"] == "openai" or model_info["type"] == "closed-model":
        response = generate_openai_answer(prompt, model_info["name"])

    return response


def process_single_query(row, model_id: str, loaded_models: Dict):
    query = row["query"]

    # Collect top3 contexts
    top3_contexts = []
    for i in range(1, 4):
        col_name = f"top_{i}_text"
        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip():
            top3_contexts.append(str(row[col_name]).strip())

    if not top3_contexts:
        return None

    rag_answer = generate_rag_answer(query, top3_contexts, model_id, loaded_models)

    return {
        "query": query,
        "model_name": loaded_models[model_id]["name"],
        "rag_answer": rag_answer,
        "top3_contexts": top3_contexts,
    }


def process_csv_generation(
    input_file_path: str,
    base_output_dir: str,
    selected_models: List[str],
    max_rows=None,
):
    df = pd.read_csv(input_file_path)

    if max_rows is not None:
        df = df.head(max_rows)

    loaded_models = load_models(selected_models)

    print(f"Processing {len(df)} queries with {len(selected_models)} models")

    # Process each model separately and save to individual files
    for model_id in selected_models:
        if model_id not in loaded_models:
            continue

        model_name = loaded_models[model_id]["name"]
        safe_model_name = (
            model_name.replace("/", "_").replace("-", "_").replace(" ", "_")
        )
        output_file_path = os.path.join(
            base_output_dir, f"rag_generation_{safe_model_name}.csv"
        )

        print(f"\nProcessing model: {model_name}")
        print(f"Output will be saved to: {output_file_path}")

        all_results = []

        for index, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {model_name}"
        ):
            result = process_single_query(row, model_id, loaded_models)
            if result:
                all_results.append(result)
            time.sleep(0.5)

            # Save progress every 10 rows
            if (index + 1) % 10 == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(f"{output_file_path}.temp", index=False)

        if all_results:
            results_df = pd.DataFrame(all_results)

            # Clean text columns for CSV output
            text_columns = ["query", "rag_answer"]
            for col in text_columns:
                if col in results_df.columns:
                    results_df[col] = results_df[col].apply(
                        lambda x: (
                            f'"{str(x).replace(chr(10), " ").replace(chr(13), " ").strip()}"'
                            if pd.notna(x)
                            else '""'
                        )
                    )

            results_df.to_csv(output_file_path, index=False, escapechar="\\")

            # Clean up temp file
            temp_file = f"{output_file_path}.temp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

            print(f"Results for {model_name} saved to: {output_file_path}")
        else:
            print(f"No results generated for {model_name}")


def select_models_interactive():
    print("Available models:")
    for key, value in HF_MODELS.items():
        print(f"{key}: {value}")

    print("\nOpenAI models:")
    for key, value in OPENAI_MODELS.items():
        print(f"{key}: {value}")

    choice = input("Model selection (comma-separated): ").strip()
    selected = [x.strip() for x in choice.split(",")]
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Generation with Multiple Models")
    parser.add_argument(
        "--models", type=str, help="Comma-separated model IDs (e.g., 1,3,7,8)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/output-top3.csv",
        help="Path to input CSV with top3 context",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="WeatherArchive_Assessment/output",
        help="Directory to save output CSV files",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Optional: max rows to process"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.models:
        selected_models = [x.strip() for x in args.models.split(",")]
    else:
        selected_models = select_models_interactive()

    print(f"Selected models: {selected_models}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    process_csv_generation(
        input_file_path=args.input_path,
        base_output_dir=args.output_dir,
        selected_models=selected_models,
        max_rows=args.max_rows,
    )

    print(f"RAG generation completed. Results saved to: {args.output_dir}")
