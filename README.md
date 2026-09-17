# wordbyword

A chat-based Spanish tutor. You talk to a locally-hosted LLM (via
[Ollama](https://ollama.com)) whose vocabulary is steered by a per-word "word
bank": every word you've been exposed to has a mastery score that decays over
time and gets reinforced (or penalized) based on how you interact with it -
mainly whether you hover to translate it. Hover any word, in the agent's
replies or in your own draft, to see a translation; hover the 🌐 at the end of
a message to translate the whole thing at once.

## Architecture

```
apps/server   Python (FastAPI) - word bank + RL-style weighting, chat
              orchestration (talks to Ollama), translation (spaCy + Argos
              Translate), SQLite storage. All in one process.
apps/web      React + Vite + TypeScript - chat UI, hover tooltips, the
              draft-input overlay that flags English words you type.
              API types are generated from the backend's OpenAPI schema.
```

The backend only depends on Ollama's `/api/chat` HTTP contract, not any
specific model - pull whatever chat-capable model you like.

## Prerequisites

- Python 3.11+
- Node 20+
- [Ollama](https://ollama.com), running locally with a model pulled:
  ```
  ollama pull llama3.1:8b   # or any other chat-capable model
  ollama serve              # usually already running as a background service
  ```

## Setup

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
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where the server calls Ollama's chat API |
| `OLLAMA_MODEL` | `llama3.1:8b` | model name passed to Ollama |
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
see `app/wordbank/scoring.py`) and asks the model to prefer them, plus a
capped number of brand-new words. Everything the model actually says gets
lemmatized and recorded regardless, so nothing slips through untracked.

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

## Tests

```bash
cd apps/server
source .venv/bin/activate
python -m pytest
```

Covers the scoring/weighting math (`test_scoring.py`), Spanish
lemmatization (`test_lemmatizer.py`), and the dictionary-first/MT-fallback
translation logic (`test_translate_service.py`).
