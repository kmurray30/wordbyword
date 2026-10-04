# wordbyword

A chat-based Spanish tutor. You talk to an LLM, and every word you've been
exposed to has a mastery score that decays over time and gets reinforced (or
penalized) based on how you interact with it - mainly whether you hover to
translate it. Hover any word (or short group of words), in the agent's
replies or in your own draft, to see its translation in context; click the
🌐 at the end of a message to toggle a full translation underneath it.

## Architecture

```
apps/server   Python (FastAPI) - word bank exposure/familiarity tracking,
              chat orchestration (talks to OpenAI), translation (spaCy for
              lemmatizing/word-bank tracking, the LLM itself for every
              actual translation - single words, the learner's in-progress
              draft, and whole messages), SQLite storage.
apps/web      React + Vite + TypeScript - chat UI, hover tooltips, the
              draft-input overlay that flags English words you type. API
              types are generated from the backend's OpenAPI schema.
```

## Prerequisites

- Python 3.11+
- Node 20+
- An OpenAI API key

## Setup

### Backend

```bash
cd apps/server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

Environment variables (all optional except the API key, sensible defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | API key for the chat backend; chat/translation calls return 503 if unset |
| `OPENAI_CHAT_MODEL` | `gpt-6-luna` | model id sent to the API for chat replies (and coaching) - change this if that id isn't valid for your account |
| `OPENAI_TRANSLATE_MODEL` | same as `OPENAI_CHAT_MODEL` | model id used for the pure translation/glossing calls (whole-message translation, per-word hover glosses, the learner-input interpreter) - override with a smaller/faster model if translation latency matters more than matching the chat model's quality for these short, simple tasks |
| `MAX_REPLY_TOKENS` | `300` | hard cap on tokens per generation (chat + translation calls alike) |
| `WORDBYWORD_DATA_DIR` | `apps/server/data` | where the SQLite DB file lives |
| `DEEPINFRA_API_TOKEN` | *(none)* | DeepInfra API key for text-to-speech; `/tts/speak` returns 503 if unset |
| `DEEPINFRA_TTS_MODEL` | `hexgrad/Kokoro-82M` | DeepInfra model used to synthesize speech |
| `DEEPINFRA_TTS_VOICE` | *(none)* | force one specific voice regardless of language - normally left unset so the voice is picked per-request from `app/tts/kokoro_voices.py`'s language map |

### Frontend

```bash
cd apps/web
npm install
npm run dev   # http://localhost:5173
```

The page won't render past a loading screen until the backend reports its
chat API is ready - it polls `GET /health/llm` and holds the whole UI back
until that reports ready, rather than let chat/translation/hover all
separately break in their own confusing ways if you open the page before
the backend has finished starting (or, on Railway, while a serverless
backend is waking from sleep). If you see it stuck, check that the backend
is running and `OPENAI_API_KEY` is actually set.

If you change the backend's API shape (new/changed routes or schemas),
regenerate the typed client with the backend running:

```bash
npm run gen:api-types   # reads http://localhost:8000/openapi.json -> src/api/schema.ts
```

## Chat history

Reloading the page restores your prior conversation instead of starting
blank - the frontend fetches `GET /chat/history?session_id=...` on mount
and renders it before you send anything new. "Clear chat" in the header
(shown once there's a conversation to clear) calls `POST
/chat/history/clear?session_id=...`, which deletes only that session's
messages - the word bank isn't touched, so vocabulary progress survives a
cleared chat. Messages restored from history skip the passive-exposure
timer and the eager whole-message-translation fetch (`ChatMessage.tsx`'s
`fromHistory` flag) - both already fired for real the first time each
message was shown, so replaying them on every reload would double-count
familiarity and re-run LLM translation calls for nothing.

## How word tracking works

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

See `app/wordbank/scoring.py` for the decay/boost math and
`app/wordbank/store.py` for how reward events (hover, passive exposure,
typing) get applied. This is pure tracking, not generation steering - the
chat model isn't nudged toward specific words; everything it says just gets
lemmatized and recorded so familiarity stays up to date.

## Whole-message translation

Click the 🌐 on a message to toggle a translation row underneath its bubble
(better word order than stitching together the per-word glosses). It's a
toggle, not a hover popover, on purpose - an earlier hover-triggered version
stayed open forever once the translation arrived, since "is the mouse still
over it" and "has a translation been fetched" were conflated into one state.
A click you control is unambiguous, and it also gives a way to lazily fetch
a translation for a history-hydrated message, which skips the eager
prefetch below.

Every translation in this app is LLM-based (`app/translate/
llm_translate.py`). `POST /translate/text` asks the same chat model
(`app/chat/openai_client.py`) to translate, with a system prompt that asks
for the translation only, no commentary. That's a real network call, so
the frontend doesn't wait for the toggle to ask for it: `ChatMessage.tsx`
fires the request automatically as soon as a message is shown, in the
background, so it's normally already cached by the time anyone opens it.
Either way it never blocks the message itself from rendering.

**The agent's messages get one translation row (Spanish -> English).** The
**user's own messages get two** - an English row and a Spanish row - because
a learner's own typed text is often a mix of both languages with grammar
mistakes in either (e.g. "Do you hablo the espanol good?"), so a single
literal pass in one direction doesn't make sense. `POST /translate/interpret`
(`llm_translate.interpret_user_input`) asks the LLM to infer what you meant
across that mix and reply with a corrected line in each language; if it
doesn't follow that format, the route falls back to a plain `translate_text`
call for whichever line is missing
rather than showing nothing. The English row has a small pencil icon since
that guess is exactly the part most likely to need a correction - editing it
re-derives the Spanish row (and its audio) from your correction instead of
the original text. Every row past the raw user input (both of the user's
rows, plus the agent's) gets its own 🔊 button, reusing the same TTS
pipeline as the agent's raw-message audio; the raw user input itself doesn't,
since its grammar or spelling might be exactly what's in question.

**Hovering a word (or short group of words) in the input box** gives its
in-context translation via `POST /translate/tag-input`, and clicking a
clickable one replaces that span in place. Fired after every word boundary
while typing (debounced), this asks the LLM to gloss the whole draft in one
call (`llm_translate.tag_draft`) - the same `{translation, spans}` shape
`gloss_reply` already uses for the agent's own replies, just run on the
learner's still-being-typed, possibly mixed-language text instead. The
model decides span boundaries itself (usually one word, occasionally a few
words grouped together for an idiom or phrasal verb) and, since it has real
sentence context, picks the one sense that applies rather than needing a
Spanish/English toggle. Each span's surface text is matched back to exact
character offsets by `app/translate/span_matching.py` - a span the model
hallucinated or couldn't be located in the draft is just dropped, not
misplaced.

## Text-to-speech

Click the 🔊 next to an assistant message to hear it spoken aloud, so you can
hear correct pronunciation alongside the hover translations. `POST
/tts/speak` (`app/tts/deepinfra_client.py`) calls DeepInfra's OpenAI-
compatible `/v1/audio/speech` endpoint, using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(`hexgrad/Kokoro-82M` on DeepInfra), and streams the mp3 back; the frontend
plays it via the browser's `Audio` API and caches the blob per message so
replaying doesn't re-fetch. This is a separate hosted API from the chat
model above - it needs its own `DEEPINFRA_API_TOKEN` (see env vars).

Kokoro doesn't infer language from the input text - each of its 54 voices is
tied to one language (its phonemizer rules follow the voice, not the text),
so the request body includes a `language` field (`es` by default, matching
`TARGET_LANGUAGE`) and `app/tts/kokoro_voices.py` maps it to the right voice:
the full 9-language, 54-voice catalog Kokoro ships with, plus one default
voice per language (`voice_for_language`). Spanish text uses `ef_dora`.
`DEEPINFRA_TTS_VOICE` overrides the map entirely if you want to force one
specific voice regardless of language.

The "Voice" picker in the header lets you choose between Spanish's three
voices - Dora (`ef_dora`), Alex (`em_alex`), Santa (`em_santa`) - fetched
from `GET /tts/voices?language=es` and persisted in `localStorage`, and
never any of Kokoro's other languages' voices. `POST /tts/speak` takes an
optional `voice` field and rejects one that isn't valid for the given
`language` with a 400.

## Deployed on Railway

Two services in one Railway project:

| Service | What | Public? |
|---|---|---|
| `server` | `apps/server` - FastAPI backend | Yes - `web` calls it over its public domain |
| `web` | `apps/web` - static Vite build | Yes |

To redeploy: push to the branch each service tracks - Railway rebuilds
automatically on every push that touches that service's directory.

## Known limitations (prototype scope)

- **spaCy's small Spanish model (`es_core_news_sm`) occasionally mis-tags
  uncommon conjugated forms** (its POS tagger sometimes marks a preterite verb
  as an adjective, which skips lemmatization for that token). Swap in
  `es_core_news_md` for better accuracy at the cost of a larger download.
- **English-vs-Spanish detection for typed input** is a heuristic (checks a
  bundled ~300-word Spanish list plus whether spaCy successfully inflected the
  token), not a real language classifier - uncommon correctly-typed Spanish
  words may occasionally get flagged as "unknown."
- **Single user, no auth, but per-browser chat history.** There's still one
  shared word bank in one SQLite file (vocabulary mastery is meant to
  persist regardless of device), but each browser gets its own chat
  history thread via a random id generated client-side and kept in
  `localStorage` (`app/lib/session.ts`) - not real session/auth, just
  enough to stop every visitor's conversation from landing in one shared
  thread (`apps/web/src/lib/session.ts`). No server-side session expiry; a
  thread lives until its "Clear chat" button is used.
- **Every chat turn makes at least two OpenAI calls** (one to generate the
  reply, one to gloss/translate it), and the hover-translate features add
  more on top - fine at this app's traffic level, but worth knowing if you
  scale up usage, since it's a pay-per-token hosted API now rather than a
  self-hosted model sitting idle between turns.
- Runs as a **web app**; the longer-term plan is to port the UI to React
  Native. Because the backend is a plain HTTP/JSON API, that port doesn't
  require backend changes - RN talks to it the same way the web app does.

## Tests

```bash
cd apps/server
source .venv/bin/activate
python -m pytest
```

Covers the familiarity/decay math (`test_scoring.py`), Spanish
lemmatization (`test_lemmatizer.py`), the LLM-backed translation/glossing
helpers (`test_llm_translate.py`), and the chat/translate routes
(`test_chat_route.py`, `test_tag_input.py`), among others.
