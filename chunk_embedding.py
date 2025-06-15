import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize
import tiktoken

# Localize Chroma
client = chromadb.PersistentClient(path="weather_chroma_store")
collection = client.get_or_create_collection(name="weather_records")

# Read database csv file (just first 2 data)
df = pd.read_csv(r"C:\Users\14821\Desktop\RAG\MixedCTX_Dataset(1386).csv")
subset = df.iloc[:2]

# Initialize tokenizer (ChatGPT tokenizer)
tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

# sliding window chunking
def chunk_text(text, max_tokens=125, overlap=20):
    sentences = sent_tokenize(text, language="english")
    chunks = []
    current_chunk = []
    current_len = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        token_len = len(tokenize(sentence))

        # If adding the sentence exceeds the max token limit
        if current_len + token_len > max_tokens:
            if current_chunk:
                # Finalize current chunk
                chunk_text_str = " ".join(current_chunk)
                chunks.append(chunk_text_str)

                # Create new chunk with overlap (from previous chunk's end)
                if overlap > 0:
                    overlap_tokens = 0
                    new_chunk = []
                    for sent in reversed(current_chunk):
                        sent_tokens = len(tokenize(sent))
                        if overlap_tokens + sent_tokens > overlap:
                            break
                        new_chunk.insert(0, sent)
                        overlap_tokens += sent_tokens
                    current_chunk = new_chunk
                    current_len = sum(len(tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            else:
                # Very long single sentence (skip or force chunk)
                chunks.append(sentence)
                i += 1
                continue
        else:
            current_chunk.append(sentence)
            current_len += token_len
            i += 1

    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Default embedding model
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Process each article
for i, row in subset.iterrows():
    article = str(row["Article"])
    chunks = chunk_text(article)
    metadata = {"date": str(row["Date"]), "weather": row["Weather_Type"]}

    # Store each chunk into Chroma
    for j, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{row['ID']}_chunk{j}"],
            metadatas=[metadata]
        )

print("Successfully write into chroma. Total data size is:", collection.count())
