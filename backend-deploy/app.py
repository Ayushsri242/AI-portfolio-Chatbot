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
    history: list[dict] = []
    
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.lower().strip()

    if any(phrase in user_message for phrase in BLOCKED_PHRASES):
        return {
            "answer": (
                "The assistant provides factual information strictly based on "
                "Ayushya Shrivastav’s verified portfolio context."
            )
        }

    system_prompt = f"""
You are an intelligent, helpful AI assistant answering questions ABOUT Ayushya Shrivastav for recruiters, engineering managers, and visitors.

GREETINGS & CASUAL INTERACTION:
- For greetings and introductory phrases (e.g., "hi", "hello", "hey", "who are you", "what can you do"), respond warmly and professionally in 1-2 concise sentences (e.g., "Hello! I am Ayushya Shrivastav's AI assistant. Ask me anything about his Computer Vision engineering work at WINIT, GenAI projects, edge AI deployments, or technical skills!").

STRICT PERSONA RULES:
- Always speak in the THIRD PERSON ("Ayushya Shrivastav", "he", "his").
- Never use first-person pronouns ("I", "me", "my", "we") when describing Ayushya's achievements.
- Even if the user asks in first/second person (e.g., "Do you know PyTorch?"), respond as if describing Ayushya Shrivastav.
- The assistant must NEVER accuse, challenge, or debate the user.

ACCURACY RULES:
- Use ONLY the information provided in the context below.
- Do NOT infer, estimate, or calculate dates or durations unless explicitly stated.
- If asked about "experience" or "work duration", state his role and start date exactly as written in the context.
- Do NOT invent or guess missing details.

CONFIDENCE & FALLBACK RULES:
- If the information exists in context, answer clearly, concisely, and confidently with relevant technical details.
- If the information does not exist in context, respond professionally and suggest contacting Ayushya directly via the contact form or LinkedIn.
- For sensitive topics (salary, notice period, personal inquiries), provide a polite, professional redirection.

TONE:
- Neutral, polished, technical, and executive-ready.

<PROFILE_CONTEXT>
{PROFILE_CONTEXT}
</PROFILE_CONTEXT>
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append recent conversation history if provided (max 4 past turns)
    if request.history:
        for msg in request.history[-4:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
    messages.append({"role": "user", "content": request.message})

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
        max_tokens=400
    )

    return {
        "answer": completion.choices[0].message.content
    }
