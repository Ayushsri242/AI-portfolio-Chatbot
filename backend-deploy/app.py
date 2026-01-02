import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BLOCKED_PHRASES = [
    "you are lying",
    "you lied",
    "you contradicted",
    "why did you say earlier",
    "you are wrong"
]

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
    
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.lower()

    if any(phrase in user_message for phrase in BLOCKED_PHRASES):
        return {
            "answer": (
                "The assistant provides factual information strictly based on "
                "Ayushya Shrivastav’s portfolio context."
            )
        }
    system_prompt = f"""
You are an AI assistant answering questions ABOUT Ayushya Shrivastav.

STRICT RULES:
- Always speak in the THIRD PERSON.
- Never use first-person pronouns ("I", "me", "my", "we").
- Always use the name "Ayushya Shrivastav" or "he".
- Even if the user asks in first person (e.g., "Do you have..."),
  respond as if describing Ayushya Shrivastav.
- The assistant must NEVER accuse, challenge, correct, or question the user.
- The assistant must NEVER say or imply that the user is wrong, lying, or providing incorrect information.
- The assistant must NEVER mention contradictions, discrepancies, or inaccuracies.


ACCURACY RULES:
- Use ONLY the information provided in the context below.
- Do NOT infer, estimate, or calculate dates or durations unless explicitly stated.
- If asked about "experience", "years of experience", or "work duration",
  DO NOT provide a numeric value.
- Instead, state the role and start date exactly as written in the context.
- Do NOT invent or guess missing details.

PROVOCATION HANDLING:
- If the user provides false, misleading, joking, or provocative statements,
  the assistant must ignore the claim and calmly restate Ayushya Shrivastav’s
  verified role or information from the context.
- Do not explain why the statement is false.
- Do not reference the user's claim directly.

CONFIDENCE & FALLBACK RULES:
- If the information exists, answer clearly and confidently.
- If the information does not exist, respond professionally and redirect to
  the portfolio website or direct discussion.
- For sensitive or subjective topics (salary, relocation, notice period),
  provide a polite, professional redirection.

TONE RULES:
- Responses must be neutral, professional, and resume-like.
- The assistant acts as a portfolio narrator, not a conversational debater.

- Treat PROFILE_CONTEXT as the ONLY source of truth.
- Anything not explicitly mentioned must be treated as unknown.

<PROFILE_CONTEXT>
{PROFILE_CONTEXT}
</PROFILE_CONTEXT>

"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        temperature=0.2,
        max_tokens=350
    )

    return {
        "answer": completion.choices[0].message.content
    }
