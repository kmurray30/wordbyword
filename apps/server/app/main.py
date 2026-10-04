import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import OPENAI_API_KEY
from app.db import init_db
from app.routes import chat, events, translate, tts
from app.schemas import LlmHealthResponse

app = FastAPI(title="wordbyword", version="0.1.0")

_default_origins = "http://localhost:5173"
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(translate.router)
app.include_router(events.router)
app.include_router(tts.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/llm", response_model=LlmHealthResponse)
def health_llm() -> LlmHealthResponse:
    """Whether the chat backend (OpenAI) is actually configured and able to
    serve a completion right now - distinct from /health above, which only
    means this process is up. Ready as soon as a key is configured; the
    frontend polls this on page load and holds the UI back until it reports
    ready, which is also what wakes Railway's own serverless sleep on the
    backend process itself, if it's asleep."""
    return LlmHealthResponse(ready=bool(OPENAI_API_KEY))
