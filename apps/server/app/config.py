import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WORDBYWORD_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("WORDBYWORD_DB_URL", f"sqlite:///{DATA_DIR / 'wordbyword.db'}")

# llama-server (llama.cpp), not Ollama - see app/chat/llama_client.py for why.
# Local dev: run llama-server yourself (see README). In Railway, this points
# at the model-server service over private networking.
MODEL_SERVER_BASE_URL = os.environ.get("MODEL_SERVER_BASE_URL", "http://localhost:8080")

# Pinned to the specific model deployed to model-server (apps/model-server/Dockerfile)
# so the tokenizer used for logit_bias matches the model actually generating text.
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")

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

# Word-bank / RL weighting tuning knobs.
# Kept modest (was 2/6): Qwen3-0.6B is small enough that forcing many words
# into one short reply via logit_bias tends to produce ungrammatical output
# rather than a natural sentence - fewer, gentler nudges per turn holds up
# better than more aggressive ones on a model this size.
NEW_WORDS_PER_TURN = 1
REINFORCE_WORDS_PER_TURN = 3
DEFAULT_REVIEW_INTERVAL_DAYS = 1.0
MIN_REVIEW_INTERVAL_DAYS = 0.25
MAX_REVIEW_INTERVAL_DAYS = 60.0
PASSIVE_EXPOSURE_BOOST = 0.05
ACTIVE_RECALL_BOOST = 0.15
HOVER_PENALTY = 0.30
INTERVAL_GROWTH_ON_SUCCESS = 1.6
INTERVAL_SHRINK_ON_FAILURE = 0.5
