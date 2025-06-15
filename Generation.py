from Retrieve import hybrid_retrieve, bm25_model, collection
import openai
openai.api_key = "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"

query = "Rocky mountains' climate"

# Get top 5 chunks
top_docs = hybrid_retrieve(query, bm25_model, collection, top_k=5)

# Combine the paragraphs
context = "\n\n".join([doc for _, _, doc, _, _ in top_docs])

prompt = f"""Use only the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

# LLM generate anwser
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print("GPT：", response.choices[0].message.content)
