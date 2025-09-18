# backend/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
import os, requests, re
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})

# --- DB models ---
class JournalEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    mood: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MoodEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    mood: str
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Pydantic request models ---
class JournalCreate(BaseModel):
    text: str
    mood: Optional[str] = None

class MoodCreate(BaseModel):
    mood: str
    note: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class QuizRequest(BaseModel):
    answers: List[int]  # expected 9 ints (0-3)

# --- App setup ---
app = FastAPI(title="Student Mental Wellness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- Utility: crisis detection ---
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "hurt myself", "want to die", "cant go on", "can't go on"
]

def detect_crisis(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    for kw in CRISIS_KEYWORDS:
        if kw in t:
            return True
    return False

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def call_openai_chat(user_message: str) -> Optional[str]:
    if not OPENAI_KEY:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "You are an empathetic, non-judgmental assistant that offers supportive, general mental wellness "
        "suggestions for students. You are NOT a therapist and must always include a brief reminder to seek "
        "professional help for serious issues. Keep recommendations practical and culturally sensitive."
    )
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

# --- Startup ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- Routes ---
@app.get("/api/health")
def health():
    return {"status": "ok"}

# Journals
@app.post("/api/journal")
def create_journal(item: JournalCreate, session: Session = Depends(get_session)):
    je = JournalEntry(text=item.text, mood=item.mood)
    session.add(je)
    session.commit()
    session.refresh(je)
    return je

@app.get("/api/journal")
def list_journal(session: Session = Depends(get_session)):
    entries = session.exec(select(JournalEntry).order_by(JournalEntry.created_at.desc())).all()
    return entries

# Mood
@app.post("/api/mood")
def create_mood(item: MoodCreate, session: Session = Depends(get_session)):
    me = MoodEntry(mood=item.mood, note=item.note)
    session.add(me)
    session.commit()
    session.refresh(me)
    return me

@app.get("/api/mood")
def list_mood(session: Session = Depends(get_session)):
    entries = session.exec(select(MoodEntry).order_by(MoodEntry.created_at.desc())).all()
    return entries

# Quick PHQ-9 style quiz (9 answers 0-3)
@app.post("/api/quiz")
def submit_quiz(q: QuizRequest):
    if len(q.answers) != 9:
        raise HTTPException(status_code=400, detail="Provide 9 answers (0-3 each).")
    if any(not isinstance(a, int) or a < 0 or a > 3 for a in q.answers):
        raise HTTPException(status_code=400, detail="Answers must be integers 0..3.")
    score = sum(q.answers)
    if score <= 4:
        interp = "Minimal or none"
    elif score <= 9:
        interp = "Mild"
    elif score <= 14:
        interp = "Moderate"
    elif score <= 19:
        interp = "Moderately severe"
    else:
        interp = "Severe"
    advice = (
        "This is not a diagnosis. If your score is 10 or above consider seeking professional help. "
        "If you are in immediate danger call local emergency services."
    )
    return {"score": score, "interpretation": interp, "advice": advice}

# Chat endpoint: tries OpenAI (if key present) else fallback; detects crisis keywords
@app.post("/api/chat")
def chat(req: ChatRequest):
    if detect_crisis(req.message):
        # Immediate crisis response (non-therapeutic)
        return {
            "reply": (
                "I'm really sorry you're feeling this way. If you're in immediate danger, please call local emergency services now. "
                "If you can, reach out to a trusted person nearby. For professional support, contact a local mental health helpline. "
                "This service is not a substitute for urgent professional care."
            )
        }
    # Try OpenAI
    if OPENAI_KEY:
        ai_reply = call_openai_chat(req.message)
        if ai_reply:
            return {"reply": ai_reply}
    # Fallback simple supportive responses
    msg = req.message.lower()
    if any(w in msg for w in ["stress", "stressed", "exam", "exams", "pressure"]):
        return {"reply": "Exams and pressure can be very stressful. Try a 4-4-4 breathing break: inhale 4s — hold 4s — exhale 4s. Small breaks and talk to a friend or counselor can help."}
    if any(w in msg for w in ["anxious", "anxiety", "panic"]):
        return {"reply": "I'm sorry you're feeling anxious. Grounding exercises (name 5 things you see, 4 you can touch) can help in the moment. Consider talking to someone you trust or a counselor."}
    if any(w in msg for w in ["sad", "depressed", "down"]):
        return {"reply": "I'm sorry you're feeling low. Small steps like short walks, journaling, or reaching out to a friend can help. If these feelings persist consider professional support."}
    # Default
    return {"reply": "I hear you. Could you tell me more? If this is an emergency please contact local emergency services or a nearby trusted person."}
