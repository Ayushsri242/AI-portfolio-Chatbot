import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize app
app = FastAPI()

# Load embedding model
embedding_model = None


# Load FAISS index and data
index = faiss.read_index("vectorstore/index.faiss")

with open("vectorstore/texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("vectorstore/sources.pkl", "rb") as f:
    sources = pickle.load(f)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {"status": "AI Portfolio Chatbot is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    # Embed user query
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    query_embedding = embedding_model.encode([request.message])


    # Retrieve top-k relevant chunks
    _, indices = index.search(query_embedding, 5)
    retrieved_chunks = [texts[i] for i in indices[0]]

    context = "\n".join(retrieved_chunks)

    system_prompt = f"""
You are an AI portfolio assistant for Ayushya Shrivastav.

Rules:
- Answer ONLY using the provided context.
- If multiple items match (projects, skills, roles), list all of them.
- If dates or durations are incomplete, say so explicitly.
- If the answer is not present, say you do not have that information.


Context:
{context}

User Question:
{request.message}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": system_prompt}
        ],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources[i] for i in indices[0]))
    }
