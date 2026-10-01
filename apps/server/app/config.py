import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WORDBYWORD_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("WORDBYWORD_DB_URL", f"sqlite:///{DATA_DIR / 'wordbyword.db'}")

# llama-server (llama.cpp), not Ollama - see app/chat/llama_client.py for why.
# Local dev: run llama-server yourself (see README). In Railway, this points
# at the model-server service over private networking.
MODEL_SERVER_BASE_URL = os.environ.get("MODEL_SERVER_BASE_URL", "http://localhost:8080")

# Which chat backend app/chat/model_client.py routes to: "local" (the
# self-hosted llama-server above, the default - no per-request cost, but
# something you have to run) or "openai" (a hosted, pay-per-token API - see
# app/chat/openai_client.py; no infra to run, but costs money per call and
# can't support word-bank vocabulary steering, see WORD_WEIGHTING_ENABLED
# below). This is only the BOOT default - the live value a user actually
# gets is the UI-togglable one in app/settings_store.py (GET/PUT /settings),
# which seeds itself from this env var the first time it's ever read.
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "local").strip().lower()

# Hosted chat backend (app/chat/openai_client.py), only used when the
# active provider is "openai". Empty key means that provider is
# unconfigured - openai_client.py raises a clear error rather than failing
# obscurely, same pattern as DEEPINFRA_API_TOKEN below for TTS.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-6-luna")

# Pinned to the specific model deployed to model-server (apps/model-server/Dockerfile)
# so the tokenizer used for logit_bias matches the model actually generating text.
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen3-1.7B")

# Hard cap on tokens per generation (chat replies AND translation calls, both
# go through llama_client.chat). Without this, a generation that never hits
# a stop token runs until it exhausts the model server's context window -
# observed live: one reply generated 4000+ tokens before llama-server killed
# it with "Context size has been exceeded", which also broke every OTHER
# concurrent request sharing that context (translate calls came back stuck/
# erroring, chat turns got a spurious "could not reach the model server").
# 300 is generous for the "short, 1-3 sentence" replies/translations this
# app actually needs - it bounds the failure mode, not normal output.
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", "300"))

TARGET_LANGUAGE = "es"
NATIVE_LANGUAGE = "en"

# Kill switch for word-bank vocabulary steering - the logit_bias boost and
# the "prefer/introduce these words" lines in the system prompt. Lets you
# compare the model's raw, unsteered behavior against steered behavior
# without losing the tuning knobs below (which stay in effect the moment
# this is flipped back on). Word-bank tracking itself (exposure counts,
# hover/familiarity, review scheduling) is untouched either way - this only
# controls whether any of that gets used to steer generation. Off by
# default for now (an A/B test found even a single gently-nudged word could
# make the 1.7B model echo the question back before answering - see
# logit_bias.py); also never actually active against the "openai" provider
# regardless of this value, since logit_bias needs the local model's own
# tokenizer (see app.settings_store.weighting_active). This is only the
# BOOT default - see MODEL_PROVIDER's comment above, same caveat applies.
WORD_WEIGHTING_ENABLED = os.environ.get("WORD_WEIGHTING_ENABLED", "false").strip().lower() not in ("false", "0", "")

# Text-to-speech (DeepInfra, a hosted API - separate from the self-hosted chat
# model above). Empty token means TTS is disabled; app/tts/deepinfra_client.py
# raises a clear error rather than silently failing. Voice is picked per
# request from app/tts/kokoro_voices.py's language map, not a fixed default -
# DEEPINFRA_TTS_VOICE below is only an escape hatch to force one voice
# regardless of language.
DEEPINFRA_API_TOKEN = os.environ.get("DEEPINFRA_API_TOKEN", "")
DEEPINFRA_TTS_MODEL = os.environ.get("DEEPINFRA_TTS_MODEL", "hexgrad/Kokoro-82M")
DEEPINFRA_TTS_VOICE = os.environ.get("DEEPINFRA_TTS_VOICE", "")

# Word-bank / RL weighting tuning knobs.
# Lowered further (was 1/3) after an A/B test against WORD_WEIGHTING_ENABLED
# =false: with weighting on, replies twice echoed the question back before
# (barely) answering it - the same failure mode we upgraded 0.6B->1.7B to
# fix in the first place - while weighting-off replies answered directly.
# Forcing 4 words into a short reply every turn was likely still too much
# for a 1.7B model to do while also staying on-topic. See also
# logit_bias.py's bias magnitudes, lowered alongside this.
NEW_WORDS_PER_TURN = 0
REINFORCE_WORDS_PER_TURN = 1
DEFAULT_REVIEW_INTERVAL_DAYS = 1.0
MIN_REVIEW_INTERVAL_DAYS = 0.25
MAX_REVIEW_INTERVAL_DAYS = 60.0
PASSIVE_EXPOSURE_BOOST = 0.05
ACTIVE_RECALL_BOOST = 0.15
HOVER_PENALTY = 0.30
INTERVAL_GROWTH_ON_SUCCESS = 1.6
INTERVAL_SHRINK_ON_FAILURE = 0.5
