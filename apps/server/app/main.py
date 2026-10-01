import os

import httpx
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import settings_store
from app.chat.logit_bias import warm_up as warm_up_tokenizer
from app.config import MODEL_SERVER_BASE_URL, OPENAI_API_KEY
from app.db import get_session, init_db
from app.routes import chat, events, settings, translate, tts
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
app.include_router(settings.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    warm_up_tokenizer()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/llm", response_model=LlmHealthResponse)
def health_llm(session: Session = Depends(get_session)) -> LlmHealthResponse:
    """Whether the active chat backend can actually serve a completion
    right now - distinct from /health above, which only means this process
    is up. Under the "openai" provider there's no self-hosted model server
    to wait on at all - ready as soon as a key is configured to call it
    with, no cold-start wait to gate the UI on. Under "local" (the
    default): model-server runs with Railway's serverless mode (sleeps
    after ~5-10min idle) and, right after a cold start, spends a stretch
    loading the model's weights into memory before its own /health reports
    ready - either way, a chat/translate request that lands during that
    window fails with llama_client.ModelServerUnavailableError. Hitting it
    here (rather than trusting a cached flag) is itself the wake-up
    trigger for the sleep case - Railway wakes a serverless service on any
    inbound request. The frontend is expected to poll this on page load
    and hold the UI back until it reports ready, rather than let a real
    interaction be the first thing to hit that failure mode."""
    provider = settings_store.get_settings(session).model_provider
    if provider == "openai":
        return LlmHealthResponse(ready=bool(OPENAI_API_KEY))
    try:
        response = httpx.get(f"{MODEL_SERVER_BASE_URL}/health", timeout=5.0)
    except httpx.HTTPError:
        return LlmHealthResponse(ready=False)
    return LlmHealthResponse(ready=response.status_code == 200)
