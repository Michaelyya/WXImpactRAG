import chromadb
from nltk.tokenize import sent_tokenize
client = chromadb.PersistentClient(path="weather_chroma_store")

collection = client.get_collection(name="weather_records")


results = collection.get()

docs = results["documents"]
ids = results["ids"]
metadatas = results["metadatas"]

# print every chunks and check 句子完整性
for chunk, cid, meta in zip(docs, ids, metadatas):
    sentences = sent_tokenize(chunk, language="english")

    print(f"\n Chunk ID: {cid}")
    print(f" Metadata: {meta}")

    # chunk's head checking
    if not chunk.strip().startswith(sentences[0].strip()):
        print("Start of chunk is not the head of the sentence ❌")
    else:
        print("Start of chunk is the head of the sentence ✔ ")

    # chunk's end checking
    if not chunk.strip().endswith(sentences[-1].strip()):
        print("End of chunk is not the end of the sentence ❌")
    else:
        print("End of chunk is the end of the sentence ✔")

    # Print whole chunks
    print("Whole Chunks：\n" + chunk)
    print("=" * 80)