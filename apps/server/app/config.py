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
# a stop token runs until it exhausts the model's context window.
#
# Raised from 300 after a live failure: OPENAI_CHAT_MODEL is a reasoning-
# style model (confirmed by openai_client.py's own comment on requiring
# max_completion_tokens over the classic max_tokens field) whose internal
# reasoning tokens count against this SAME budget before any visible output
# - see llm_translate.py's _TAG_DRAFT_MAX_TOKENS/_GLOSS_REPLY_MAX_TOKENS/
# _COACH_TRANSLATIONS_MAX_TOKENS comments for the first time this bit:
# those calls' heavier structured-JSON prompts needed a much bigger budget
# (1600-3000) for the exact same reason. 300 mostly worked for a plain
# conversational reply (much lower reasoning overhead than a multi-part
# JSON-extraction prompt) - "mostly" being the problem: on a harder turn,
# reasoning alone could exhaust it, leaving ZERO visible content tokens.
# Over /chat/turn/stream that shows up as literally nothing: no "chunk"
# events ever arrive (the frontend's "…" placeholder never fills in), and
# the final "done" event carries an empty `text` - a visibly blank chat
# bubble, not an error. 1200 is still "generous for short replies," not a
# real ceiling on them - a short reply still stops on its own well under
# this - it just gives a harder turn's reasoning phase room to finish
# before any visible tokens are expected.
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", "1200"))

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
