import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WORDBYWORD_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("WORDBYWORD_DB_URL", f"sqlite:///{DATA_DIR / 'wordbyword.db'}")

# Hosted chat backend (app/chat/openai_client.py) - the only chat backend.
# Empty key means it's unconfigured - openai_client.py raises a clear error
# rather than failing obscurely, same pattern as DEEPINFRA_API_TOKEN below
# for TTS.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-6-luna")

# Model used for the pure translation/glossing calls (translate_text,
# gloss_reply, interpret_user_input, tag_draft - everything in
# llm_translate.py except coach_draft_stream, which mixes conversational
# feedback with translation and so stays on the main chat model). Defaults
# to OPENAI_CHAT_MODEL (no behavior change out of the box) - override with
# a smaller/faster model id if translation latency matters more than
# matching the chat model's quality for this app's short, simple spans.
OPENAI_TRANSLATE_MODEL = os.environ.get("OPENAI_TRANSLATE_MODEL", OPENAI_CHAT_MODEL)

# Hard cap on tokens per generation (chat replies AND translation calls, both
# go through openai_client.chat). Without this, a generation that never hits
# a stop token runs until it exhausts the model's context window. 300 is
# generous for the "short, 1-3 sentence" replies/translations this app
# actually needs - it bounds the failure mode, not normal output.
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", "300"))

TARGET_LANGUAGE = "es"
NATIVE_LANGUAGE = "en"

# Text-to-speech (DeepInfra, a hosted API - separate from the self-hosted chat
# model above). Empty token means TTS is disabled; app/tts/deepinfra_client.py
# raises a clear error rather than silently failing. Voice is picked per
# request from app/tts/kokoro_voices.py's language map, not a fixed default -
# DEEPINFRA_TTS_VOICE below is only an escape hatch to force one voice
# regardless of language.
DEEPINFRA_API_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_TTS_MODEL = os.environ.get("DEEPINFRA_TTS_MODEL", "hexgrad/Kokoro-82M")
DEEPINFRA_TTS_VOICE = os.environ.get("DEEPINFRA_TTS_VOICE", "")

# Word-bank exposure/familiarity tuning knobs.
DEFAULT_REVIEW_INTERVAL_DAYS = 1.0
MIN_REVIEW_INTERVAL_DAYS = 0.25
MAX_REVIEW_INTERVAL_DAYS = 60.0
PASSIVE_EXPOSURE_BOOST = 0.05
ACTIVE_RECALL_BOOST = 0.15
HOVER_PENALTY = 0.30
INTERVAL_GROWTH_ON_SUCCESS = 1.6
INTERVAL_SHRINK_ON_FAILURE = 0.5
