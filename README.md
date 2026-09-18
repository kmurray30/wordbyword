# wordbyword

A chat-based Spanish tutor. You talk to a locally-hosted LLM whose vocabulary
is steered by a per-word "word bank": every word you've been exposed to has a
mastery score that decays over time and gets reinforced (or penalized) based
on how you interact with it - mainly whether you hover to translate it. Hover
any word, in the agent's replies or in your own draft, to see a translation;
hover the 🌐 at the end of a message to translate the whole thing at once.

## Architecture

```
apps/model-server  llama.cpp's own server (llama-server), running Qwen3-0.6B.
                    Not Ollama - see "Why llama.cpp, not Ollama" below.
apps/server         Python (FastAPI) - word bank + RL-style weighting, chat
                    orchestration (talks to model-server), translation
                    (spaCy + Argos Translate), SQLite storage.
apps/web            React + Vite + TypeScript - chat UI, hover tooltips, the
                    draft-input overlay that flags English words you type.
                    API types are generated from the backend's OpenAPI schema.
```

### Why llama.cpp, not Ollama

The whole point of the word bank is to make the model **actually** more
likely to use specific words - not just be asked nicely. That requires
`logit_bias`: directly boosting a token's sampling probability during
generation. **Ollama has no `logit_bias` support** - it's an open, unresolved
feature request (`ollama/ollama#3795`, filed April 2024). Ollama can read
token probabilities (`logprobs`) but can't bias them.

`llama-server` (llama.cpp's own server binary, distinct from Ollama, which is
itself built on llama.cpp) has a stable, documented `logit_bias` parameter on
its `/v1/chat/completions` endpoint. `apps/server/app/chat/llama_client.py`
calls it with a `logit_bias` computed per turn from each due-for-review or
new word's first token id (`apps/server/app/chat/logit_bias.py`) - a real
mechanical nudge, layered on top of (not instead of) the existing
system-prompt instruction. See those two files' docstrings for the full
reasoning, including why the bias only targets each word's dictionary/base
form and not its inflected forms (a deliberate design choice, not a gap).

## Prerequisites

- Python 3.11+
- Node 20+
- Docker (to run `llama-server` locally) - or a native llama.cpp build if you
  prefer not to use Docker.

## Setup

### Model server

```bash
docker build -t wordbyword-model apps/model-server
docker run --rm -p 8080:8080 wordbyword-model
```

First run downloads the Qwen3-0.6B-GGUF weights (~650MB, Q8_0) from Hugging Face;
subsequent runs reuse them if you mount a volume at `/models` (see the
Dockerfile - `LLAMA_CACHE=/models`). Leave this running; the backend talks to
it at `http://localhost:8080` by default.

### Backend

```bash
cd apps/server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python scripts/install_translate_models.py   # installs Argos Translate en<->es packages
```

`install_translate_models.py` downloads from Argos Translate's package index,
which needs unrestricted outbound access - it will fail in network-locked
sandboxes (it does in this repo's dev container) but works fine on a normal
machine.

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

Environment variables (all optional, sensible defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SERVER_BASE_URL` | `http://localhost:8080` | where the server calls llama-server's chat API |
| `TOKENIZER_NAME` | `Qwen/Qwen3-0.6B` | HF tokenizer used to compute `logit_bias` token ids - must match the model running in model-server |
| `WORDBYWORD_DATA_DIR` | `apps/server/data` | where the SQLite DB file lives |

### Frontend

```bash
cd apps/web
npm install
npm run dev   # http://localhost:5173
```

If you change the backend's API shape (new/changed routes or schemas),
regenerate the typed client with the backend running:

```bash
npm run gen:api-types   # reads http://localhost:8000/openapi.json -> src/api/schema.ts
```

## How the word bank / weighting works

Every lemma (dictionary base form - `hablar`, not `hablando`) the agent has
ever used gets a `familiarity` score that decays over time (a forgetting
curve, using `review_interval_days` as the half-life) and moves in response to
three signals, all tied to what you actually do in the UI:

- **You don't hover a word the agent used** -> small familiarity boost
  (passive recognition), and its review interval grows - it's due for review
  further out.
- **You hover/translate a word the agent used** -> familiarity drops and the
  review interval shrinks - you'll see it again sooner.
- **You type the word yourself**, unprompted -> a bigger familiarity boost
  (active recall is stronger evidence than passive recognition).

Each chat turn, the server samples a small set of "due" words (weighted
toward low-familiarity, overdue-for-review words, but not deterministically -
see `app/wordbank/scoring.py`) and both (a) tells the model to prefer them in
the system prompt and (b) computes a `logit_bias` that directly boosts those
words' first-token sampling probability (`app/chat/logit_bias.py`), scaled by
how overdue each word is. New words get a flat bias. Everything the model
actually says gets lemmatized and recorded regardless, so nothing slips
through untracked.

## Deployed on Railway

Three services in one Railway project, wired over private networking:

| Service | What | Public? |
|---|---|---|
| `model-server` | `apps/model-server` - llama-server + Qwen3-0.6B | No - only `server` calls it, over `model-server.railway.internal:8080` |
| `server` | `apps/server` - FastAPI backend | Yes - `web` calls it over its public domain |
| `web` | `apps/web` - static Vite build | Yes |

To redeploy: push to the branch each service tracks: Railway rebuilds
automatically. `model-server`'s image only needs rebuilding if you change the
model or quantization; `server`/`web` rebuild on every push that touches
their directories.

## Known limitations (prototype scope)

- **spaCy's small Spanish model (`es_core_news_sm`) occasionally mis-tags
  uncommon conjugated forms** (its POS tagger sometimes marks a preterite verb
  as an adjective, which skips lemmatization for that token). Swap in
  `es_core_news_md` for better accuracy at the cost of a larger download.
- **English-vs-Spanish detection for typed input** is a heuristic (checks a
  bundled ~300-word Spanish list plus whether spaCy successfully inflected the
  token), not a real language classifier - uncommon correctly-typed Spanish
  words may occasionally get flagged as "unknown."
- **The new-word candidate pool** (`app/wordbank/frequency_list.py`) is a
  small hand-curated list, not a real frequency corpus.
- **Single user, no auth.** The whole app is one word bank in one SQLite file.
- Runs as a **web app**; the longer-term plan is to port the UI to React
  Native. Because the backend is a plain HTTP/JSON API, that port doesn't
  require backend changes - RN talks to it the same way the web app does.
- **`logit_bias` only targets a word's dictionary/base-form token** (`hablar`,
  not `hablo`/`hablando`/`habló`). This is intentional, not a gap: each
  inflected form is treated as its own vocabulary item to learn, not a
  variant of the lemma - so the word bank tracking a lemma across its
  conjugations (for the hover/translate/familiarity system) and the
  `logit_bias` mechanism only weighting that lemma's base form are two
  separate design choices that don't fully line up yet. Tracking surface
  forms instead of lemmas throughout would resolve that, but it's a larger
  refactor (`app/wordbank/`, `app/translate/lemmatizer.py`) left for later.

## Tests

```bash
cd apps/server
source .venv/bin/activate
python -m pytest
```

Covers the scoring/weighting math (`test_scoring.py`), Spanish
lemmatization (`test_lemmatizer.py`), and the dictionary-first/MT-fallback
translation logic (`test_translate_service.py`).
