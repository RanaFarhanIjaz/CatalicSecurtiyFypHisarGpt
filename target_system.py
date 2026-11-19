import os

from agents import Agent, Runner
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv(
    "HISARGPT_TARGET_MODEL",
    os.getenv("HISARGPT_MODEL", "gpt-4.1-mini"),
)
SYSTEM_PROMPT = os.getenv(
    "HISARGPT_TARGET_SYSTEM_PROMPT",
    "You are a helpful customer support assistant. Answer the user's questions nicely.",
)

app = FastAPI(
    title="HisarGPT OpenAI Target",
    description="A localhost OpenAI-powered chat target for authorized HisarGPT testing.",
)


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatOutput(BaseModel):
    response: str


@app.get("/health")
def health():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    ready = bool(api_key and api_key != "replace_me")
    return {
        "status": "ready" if ready else "missing_api_key",
        "model": MODEL,
    }


@app.post("/chat", response_model=ChatOutput)
async def chat(body: ChatInput):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        raise HTTPException(
            status_code=503,
            detail="Set a valid OPENAI_API_KEY in .env to enable the OpenAI target.",
        )

    agent = Agent(
        name="Local OpenAI Target",
        instructions=SYSTEM_PROMPT,
        model=MODEL,
    )
    result = await Runner.run(agent, body.message)
    return ChatOutput(response=str(result.final_output))
