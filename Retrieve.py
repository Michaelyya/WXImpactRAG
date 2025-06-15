import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
import tiktoken
import numpy as np

# Load Chroma client
client = chromadb.PersistentClient(path="weather_chroma_store")
collection = client.get_collection(name="weather_records")

disaster_prompts = [
    # Natural disaster
    "natural disaster",
    "earthquake",
    "flood",
    "hurricane",
    "tornado",
    "storm",
    "tsunami",
    "landslide",
    "wildfire",
    "volcanic eruption",

    # Climate
    "extreme weather",
    "heavy rain",
    "snowstorm",
    "hail",
    "drought",
    "heat wave",
    "cold wave",
    "weather damage",

    # Geographic related
    "mountain area",
    "coastal region",
    "river basin",
    "urban flooding",
    "rural area",
    "forest region",

    # Sup
    "emergency response",
    "evacuation",
    "rescue operation",
    "government aid",
    "disaster relief",
    "support troops",
    "civil protection",
    "humanitarian assistance"
]


embedding_func = collection._embedding_function

disaster_embeddings = embedding_func(disaster_prompts)

#calculate disaster similarity's vector cosine
def max_disaster_similarity(doc_embedding, disaster_embeddings):

    sims = []
    for d_emb in disaster_embeddings:
        # cosine similarity
        dot = np.dot(doc_embedding, d_emb)
        norm1 = np.linalg.norm(doc_embedding)
        norm2 = np.linalg.norm(d_emb)
        sims.append(dot / (norm1 * norm2 + 1e-10))
    return max(sims)


# Tokenizer and preprocess for BM25
stopwords = set(nltk.corpus.stopwords.words("english"))
def preprocess(text):# remove 1.stopwords 2.non algebra and number 3.Capital letters
    return [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stopwords]

# Load documents from Chroma and build BM25 index
results = collection.get()
documents = results["documents"]
metadatas = results["metadatas"]
ids = results["ids"]

bm25_corpus = [preprocess(doc) for doc in documents]
bm25_model = BM25Okapi(bm25_corpus)

#Hybrid retrieve function
def hybrid_retrieve(query, bm25_model, chroma_collection, top_k=5, bm25_weight=0.5, disaster_threshold = 0.3):
    # BM25 retrieval
    tokenized_query = preprocess(query)
    bm25_scores = bm25_model.get_scores(tokenized_query)

    # Semantic retrieval
    semantic_results = chroma_collection.query(
        query_texts=[query],
        n_results=len(documents)  # full similarity for alignment
    )
    semantic_docs = semantic_results["documents"][0]
    semantic_scores = semantic_results["distances"][0]  # cosine distances

    # Normalize and convert distance to similarity, because we need to uniform comparing methods(both scores are within [0,1])
    semantic_scores = [1 - s for s in semantic_scores]

    # Match by ID
    semantic_id_score_map = dict(zip(semantic_results["ids"][0], semantic_scores))

    # Get every chunk's vector
    all_doc_embeddings = chroma_collection.get(
        ids=ids,
        include=["embeddings"]
    )["embeddings"]

    # Merge: weighted sum of BM25 and semantic
    hybrid_scores = []
    for i, doc_id in enumerate(ids):
        bm25_score = bm25_scores[i]
        sem_score = semantic_id_score_map.get(doc_id, 0)
        final_score = bm25_weight * bm25_score + (1 - bm25_weight) * sem_score
        # Calculate disaster similarity and eliminate the chunks that not related to prompt
        doc_emb = np.array(all_doc_embeddings[i])
        disaster_sim = max_disaster_similarity(doc_emb, disaster_embeddings)

        if disaster_sim >= disaster_threshold:
            hybrid_scores.append((final_score, doc_id, documents[i], metadatas[i], disaster_sim))
        else:
            pass

    # Sort and return top_k
    hybrid_scores.sort(reverse=True)
    return hybrid_scores[:top_k]

# Example usage
query = "Rocky Mountains"
top_docs = hybrid_retrieve(query, bm25_model, collection, top_k=5)

for i, (score, doc_id, doc, meta, dissim) in enumerate(top_docs):
    print(f"\nRank {i+1} | Score: {score:.4f} | DisasterSim: {dissim:.4f} | ID: {doc_id} | Date: {meta['date']} | Weather: {meta['weather']}")
    print(doc[:400] + ("..." if len(doc) > 400 else ""))
    print("=" * 80)