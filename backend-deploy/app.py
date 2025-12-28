import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# Load profile context
with open("profile_context.txt", "r", encoding="utf-8") as f:
    PROFILE_CONTEXT = f.read()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    system_prompt = f"""
You are an AI assistant answering questions ABOUT Ayushya Shrivastav.

STRICT RULES:
- Always speak in the THIRD PERSON.
- Never use "I", "me", "my", "we", or "they".
- Always use the name "Ayushya Shrivastav" or "he".
- Even if the user asks in first person (e.g., "Do you have..."),
  respond as if describing Ayushya Shrivastav.

ACCURACY RULES:
- Use ONLY the information provided in the context below.
- Do NOT infer, estimate, or calculate dates or durations unless explicitly stated.
- If asked about "experience", "years of experience", or "work duration",
  DO NOT provide a numeric value.
- Instead, state the role and start date exactly as written in the context.
- Do NOT invent or guess missing details.


CONFIDENCE & FALLBACK RULES:
- If the information exists, answer clearly and confidently.
- If the information does not exist, respond professionally and redirect to
  the portfolio website or direct discussion.
- For sensitive or subjective topics (salary, relocation, notice period),
  provide a polite, professional redirection.

=== CONTEXT ===
{PROFILE_CONTEXT}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        temperature=0.2,
    )

    return {
        "answer": completion.choices[0].message.content
    }
