from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import ask_agent

app = FastAPI(title="IT Helpdesk AI Agent")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(request: ChatRequest):
    result = ask_agent(request.message)
    return {
        "question": request.message,
        "answer": result["answer"]
    }
