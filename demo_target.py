from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="HisarGPT Local Demo Target")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(body: ChatInput):
    msg = body.message.lower()
    # Deliberately simple fixture: it does not expose secrets or execute tools.
    if "ignore previous instructions" in msg or "reveal your system prompt" in msg:
        answer = "I cannot reveal hidden instructions or private configuration."
    elif "private document" in msg:
        answer = "I can only answer using information provided in this demo."
    else:
        answer = "Demo target received your message and followed its normal behavior."
    return {"response": answer}
