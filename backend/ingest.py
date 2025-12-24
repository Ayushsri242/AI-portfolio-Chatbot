import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

os.makedirs(VECTOR_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
sources = []

for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

        # simple chunking
        chunks = content.split("\n\n")
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 50:
                texts.append(chunk)
                sources.append(filename)

print(f"Total chunks created: {len(texts)}")

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

with open(os.path.join(VECTOR_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

with open(os.path.join(VECTOR_DIR, "sources.pkl"), "wb") as f:
    pickle.dump(sources, f)

print("✅ Embeddings and FAISS index created successfully")
