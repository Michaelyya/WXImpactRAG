FILE_1386_ADDRESS = "Ground-truth/MixedCTX_Dataset(1386).csv"
FILE_CANDIDATE_POOL_ADDRESS = "data/QACandidate_Pool.csv"
FILE_DESTINATION_ADDRESS = "WeatherArchive_Assessment/output"

OPENAI_TOKENIZER = "cl100k_base"
OPENAI_BASEMODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

CHROMADB_COLLECTION_NAME = "weather_records"
CHROMADB_CLIENT_ADDRESS = "weather_chroma_store"

LANGUAGE_ENGLISH = "english"

HF_MODELS = {
    "1": "meta-llama/Meta-Llama-3-8B-Instruct",
    "2": "meta-llama/Llama-3.3-70B-Instruct",
    "3": "Qwen/Qwen2.5-7B-Instruct",
    "4": "Qwen/Qwen2.5-14B-Instruct",
    "5": "Qwen/Qwen2.5-32B-Instruct",
    "6": "Qwen/Qwen2.5-72B-Instruct",
    "7": "mistralai/Ministral-8B-Instruct-2410",
    "8": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "9": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "10": "Qwen/Qwen3-4B-Instruct-2507",
}
OPENAI_MODELS = {"11": "gpt-3.5-turbo", "12": "gpt-4o", "13": "gpt-4.1"}
DEEPSEEK_MODELS = {
    "14": "DeepSeek-V3",
    "15": "claude-sonnet-4-20250514",
    "16": "claude-opus-4-1-20250805",
    "17": "gemini-2.5-pro",
}

FILE_QUERY_ADDRESS = "WeatherArchive_Retrieval/qa_pair/queries.csv"
FILE_CONCATENATED_CHUNKS_ADDRESS = (
    "WeatherArchive_Retrieval/qa_pair/concatenated_chunks.csv"
)
FILE_CORRECT_PASSAGES_ADDRESS = "WeatherArchive_Retrieval/qa_pair/correct_passages.csv"
