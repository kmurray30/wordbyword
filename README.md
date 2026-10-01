# wordbyword

A chat-based Spanish tutor. You talk to a locally-hosted LLM whose vocabulary
is steered by a per-word "word bank": every word you've been exposed to has a
mastery score that decays over time and gets reinforced (or penalized) based
on how you interact with it - mainly whether you hover to translate it. Hover
any word, in the agent's replies or in your own draft, to see a translation
(both directions - hover a Spanish word for its English meaning, an English
one for Spanish, and a cognate like "hotel" for both side by side); click the
🌐 at the end of a message to toggle a full translation underneath it.

## Architecture

```
apps/model-server  llama.cpp's own server (llama-server), running Qwen3-1.7B.
                    Not Ollama - see "Why llama.cpp, not Ollama" below.
apps/server         Python (FastAPI) - word bank + RL-style weighting, chat
                    orchestration (talks to model-server), translation
                    (spaCy for lemmatizing, a dictionary + Argos Translate
                    for single words, the LLM itself for whole-message
                    translation), SQLite storage.
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

First run downloads the Qwen3-1.7B-GGUF weights (~1.3GB, Q4_K_M) from Hugging
Face; subsequent runs reuse them if you mount a volume at `/models` (see the
Dockerfile - `LLAMA_CACHE=/models`). Leave this running; the backend talks to
it at `http://localhost:8080` by default.

### Backend

```bash
cd apps/server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python scripts/install_translate_models.py   # installs Argos Translate en<->es packages
python scripts/build_dictionary_data.py      # bundles a Wiktionary EN<->ES dictionary
```

Both scripts download from the network (Argos Translate's package index and
kaikki.org's Wiktionary exports, respectively) and need unrestricted outbound
access - they'll fail in network-locked sandboxes (as this repo's own dev
container does) but work fine on a normal machine. Neither failing is fatal
to running the app: word-level translation just falls back further down its
chain (see app/translate/service.py) - to Argos-only if the Wiktionary
dataset is missing, or to "no translation found" if Argos is missing too.

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

Environment variables (all optional, sensible defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SERVER_BASE_URL` | `http://localhost:8080` | where the server calls llama-server's chat API |
| `TOKENIZER_NAME` | `Qwen/Qwen3-1.7B` | HF tokenizer used to compute `logit_bias` token ids - must match the model running in model-server |
| `MAX_REPLY_TOKENS` | `300` | hard cap on tokens per generation (chat + translation calls alike) - see the incident note below |
| `WORDBYWORD_DATA_DIR` | `apps/server/data` | where the SQLite DB file lives |
| `DEEPINFRA_API_TOKEN` | *(none)* | DeepInfra API key for text-to-speech; `/tts/speak` returns 503 if unset |
| `DEEPINFRA_TTS_MODEL` | `hexgrad/Kokoro-82M` | DeepInfra model used to synthesize speech |
| `DEEPINFRA_TTS_VOICE` | *(none)* | force one specific voice regardless of language - normally left unset so the voice is picked per-request from `app/tts/kokoro_voices.py`'s language map |
| `WORD_WEIGHTING_ENABLED` | `false` | **boot default only** - kill switch for vocabulary steering (`logit_bias` + the prompt's "prefer/introduce these words" lines). The live value a user actually gets is the UI toggle in the app's header (GET/PUT `/settings`), which seeds itself from this on first read. Word-bank tracking (exposure, hover, familiarity) keeps working either way. Never actually active against the `openai` provider below, regardless of this value - `logit_bias` needs the local model's own tokenizer |
| `MODEL_PROVIDER` | `local` | **boot default only**, same caveat as above - which chat backend `app/chat/model_client.py` routes to: `local` (self-hosted llama-server) or `openai` (a hosted API, see below). Live-togglable from the same `/settings` UI control |
| `OPENAI_API_KEY` | *(none)* | API key for the hosted `openai` provider; chat/translation calls return 503 if that provider is selected and this is unset |
| `OPENAI_CHAT_MODEL` | `gpt-6-luna` | model id sent to the hosted API - change this if that id isn't valid for your account |

### Frontend

```bash
cd apps/web
npm install
npm run dev   # http://localhost:5173
```

The page won't render past a loading screen until the model server is
actually up - it polls `GET /health/llm` (backend -> model-server's own
`/health`) and holds the whole UI back until that reports ready, rather
than let chat/translation/hover all separately break in their own
confusing ways if you open the page before `model-server` has finished
starting. If you see it stuck, check that `model-server` (above) is
running.

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

Only open-class content words (nouns, verbs, adjectives, adverbs) are ever
picked for this - `app/wordbank/function_words.py` excludes closed-class
words (prepositions, conjunctions, determiners, pronouns) from both the
reinforce and new-word candidate pools. Those have a narrow, fixed
grammatical slot, so hard-boosting one on a small model tends to produce a
broken sentence (e.g. a reply ending in a dangling "...por qué?") rather
than a natural one - they still get tracked from ordinary exposure, just
never targeted for forced introduction. For the same reason, `logit_bias`
is deliberately mild (see the magnitudes in `logit_bias.py`) and only one
word is targeted per turn by default (`REINFORCE_WORDS_PER_TURN` /
`NEW_WORDS_PER_TURN` in `config.py`, currently 1/0) - even Qwen3-1.7B is
small enough that forcing several words into one short reply derails the
reply more than it helps review (see below).

Turn steering off entirely via the "Word weighting" toggle in the app's
own header (backed by `GET`/`PUT /settings`, live - no redeploy needed),
or `WORD_WEIGHTING_ENABLED=false` as the boot default it seeds from: no
`logit_bias`, and the system prompt's vocabulary-rules section goes empty
- while leaving all the tracking (exposure counts, hover, familiarity)
running underneath. Useful for isolating "is the model bad" from "is the
steering making the model worse" when a reply looks off.

### Tuning history: why the numbers are what they are

An A/B test (`WORD_WEIGHTING_ENABLED=false` vs `true`, several distinct
messages sampled each way against the live deploy) traced two separate bugs
that had been making replies look "dumb," neither of which turned out to be
word-bank steering being fundamentally incompatible with reply quality:

1. **Some replies came back completely empty**, under both weighting on and
   off - not a steering bug. Qwen3's chat template enables its
   `<think>...</think>` reasoning mode by default with no token cap; on a bad
   roll the model spent the whole turn "thinking" and emitted nothing after
   the closing tag. Fixed by passing `chat_template_kwargs.enable_thinking:
   false` in `llama_client.chat`'s request payload - this app has no UI for a
   reasoning trace anyway.
2. With thinking disabled, the original forcing (`REINFORCE_WORDS_PER_TURN=3`,
   `NEW_WORDS_PER_TURN=1`, `logit_bias` up to 3.0) made the model **echo the
   question back as a reworded question with zero actual answer**, on every
   sampled message - worse than the empty-reply bug it replaced. A small
   model without its reasoning scratchpad leaned much more heavily on literal
   prompt-following, and forcing several words into a reply left no room for
   one that actually answered anything.

The fix wasn't to give up on steering - it was lowering
`REINFORCE_WORDS_PER_TURN`/`NEW_WORDS_PER_TURN` to 1/0 and the `logit_bias`
magnitudes further (see `logit_bias.py`), plus rewriting the system prompt's
top instruction to include a concrete good/bad example of exactly the
echoing failure mode instead of just an abstract "don't do this" rule - small
models pattern-match a concrete example far more reliably than they follow
an abstract prohibition. Re-tested after both changes: 6/6 sampled replies
were real, on-topic, grammatically correct answers with a natural follow-up
question, no empties, no echoing. The toggle itself remains useful as a
standing diagnostic for isolating future "is the model bad" reports - it's
currently off by default (see `WORD_WEIGHTING_ENABLED` above), a separate,
later decision unrelated to this test's result.

### Incident: an unbounded reply broke the whole model server

Live symptoms that looked unrelated turned out to share one cause: a chat
reply that seemed to come out of nowhere (answering a question that was
never asked), a translation stuck forever on "Translating...", and a
`/chat/turn` request failing with "Could not reach the model server" - all
within the same few minutes. `model-server`'s own logs showed why: one
generation ran to 4000+ tokens without ever hitting a stop token, until
llama-server killed it with "Context size has been exceeded" - and since
`n_slots` share one context, that also broke every *other* request in
flight at the time (their tasks got killed as collateral damage, which is
what actually produced the stuck-translating and unreachable-server
symptoms; neither request was really stuck or the server really down).

Nothing in `llama_client.chat`'s request ever bounded how long a reply
could run - `MAX_REPLY_TOKENS` (300, `app/config.py`) fixes that. 300 is
generous for the short replies/translations this app actually needs; it
bounds the failure mode, not normal output.

## Chat backend: local vs. hosted

Every LLM-backed call (chat turns, per-word glossing, whole-message
translation, the learner-input interpreter) routes through
`app/chat/model_client.py`, which picks between two backends based on a
live, UI-togglable setting (the "Model" dropdown in the app's own header -
backed by `GET`/`PUT /settings`, persisted in SQLite, no redeploy needed
to flip it):

- **`local`** (the default) - the self-hosted llama-server described
  above (`app/chat/llama_client.py`). Free per-call, but something you
  have to run; supports word-bank vocabulary steering (`logit_bias`).
- **`openai`** - a hosted, pay-per-token API (`app/chat/openai_client.py`,
  `OPENAI_API_KEY`/`OPENAI_CHAT_MODEL` env vars). Nothing to run or keep
  warm, but costs money per call, and **cannot support word-bank
  weighting** - `logit_bias` is keyed to the local model's own tokenizer
  (see `logit_bias.py`), which is meaningless against a hosted model's own,
  different tokenizer. The "Word weighting" toggle next to the Model
  dropdown is disabled whenever this provider is active, and the backend
  independently forces it off regardless of the stored value
  (`app.settings_store.weighting_active`) - not just a UI nicety.

`OPENAI_CHAT_MODEL` defaults to `gpt-6-luna`; if that id isn't valid for
your account, override it with no code change.

## Whole-message translation

Click the 🌐 on a message to toggle a translation row underneath its bubble
(better word order than stitching together the per-word glosses). It's a
toggle, not a hover popover, on purpose - an earlier hover-triggered version
stayed open forever once the translation arrived, since "is the mouse still
over it" and "has a translation been fetched" were conflated into one state.
A click you control is unambiguous, and it also gives a way to lazily fetch
a translation for a history-hydrated message, which skips the eager
prefetch below.

Translation itself used to go through Argos Translate (still used for
single-word lookups - see `app/translate/service.py`), but its offline MT
produced rough, sometimes outright wrong translations on short/informal
Spanish. `POST /translate/text` (`app/translate/llm_translate.py`) now asks
the same local chat model (`llama_client.chat`) to translate instead, with a
system prompt that asks for the translation only, no commentary. The LLM is
slower per call than Argos was, so the frontend doesn't wait for the toggle
to ask for it: `ChatMessage.tsx` fires the request automatically as soon as
a message is shown, in the background, so it's normally already cached by
the time anyone opens it. Either way it never blocks the message itself
from rendering.

**The agent's messages get one translation row (Spanish -> English).** The
**user's own messages get two** - an English row and a Spanish row - because
a learner's own typed text is often a mix of both languages with grammar
mistakes in either (e.g. "Do you hablo the espanol good?"), so a single
literal pass in one direction doesn't make sense. `POST /translate/interpret`
(`llm_translate.interpret_user_input`) asks the LLM to infer what you meant
across that mix and reply with a corrected line in each language; if it
doesn't follow that format (small local models don't always), the route
falls back to a plain `translate_text` call for whichever line is missing
rather than showing nothing. The English row has a small pencil icon since
that guess is exactly the part most likely to need a correction - editing it
re-derives the Spanish row (and its audio) from your correction instead of
the original text. Every row past the raw user input (both of the user's
rows, plus the agent's) gets its own 🔊 button, reusing the same TTS
pipeline as the agent's raw-message audio; the raw user input itself doesn't,
since its grammar or spelling might be exactly what's in question.

**Hovering a word in the input box** (not just the chat bubbles) gives a
ranked list of translation candidates via `POST /translate/tag-input`, and
clicking one replaces that word in place. Every real word is checked
independently against both languages' dictionaries
(`app/translate/word_validity.py`, backed by `pyspellchecker`'s bundled
offline word-frequency dictionaries), not classified into a single
Spanish-or-English bucket: a word valid Spanish gets an unclickable EN gloss
column (it's already correct - there's nothing to replace), a word valid
English gets a clickable ES translation column (swaps it in place), and a
word valid in *both* - like "once" (Spanish for "eleven", also an English
word) or "hotel" - gets both columns side by side, independently. A word
neither dictionary recognizes (a typo, a name, slang) falls back to a single
best-guess column from spaCy's morphology.

## Text-to-speech

Click the 🔊 next to an assistant message to hear it spoken aloud, so you can
hear correct pronunciation alongside the hover translations. `POST
/tts/speak` (`app/tts/deepinfra_client.py`) calls DeepInfra's OpenAI-
compatible `/v1/audio/speech` endpoint, using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(`hexgrad/Kokoro-82M` on DeepInfra), and streams the mp3 back; the frontend
plays it via the browser's `Audio` API and caches the blob per message so
replaying doesn't re-fetch. This is a separate hosted API from the
self-hosted chat model above - it needs its own `DEEPINFRA_API_TOKEN` (see
env vars) and has no Ollama-style local-only fallback.

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

Three services in one Railway project, wired over private networking:

| Service | What | Public? |
|---|---|---|
| `model-server` | `apps/model-server` - llama-server + Qwen3-1.7B | No - only `server` calls it, over `model-server.railway.internal:8080` |
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
- **Single user, no auth, but per-browser chat history.** There's still one
  shared word bank in one SQLite file (vocabulary mastery is meant to
  persist regardless of device), but each browser gets its own chat
  history thread via a random id generated client-side and kept in
  `localStorage` (`app/lib/session.ts`) - not real session/auth, just
  enough to stop every visitor's conversation from landing in one shared
  thread (`apps/web/src/lib/session.ts`). No server-side session expiry; a
  thread lives until its "Clear chat" button is used.
- **Whole-message translation doubles the load on model-server per turn**
  (one call to generate the reply, one to translate it) since both now go
  through the same small local LLM. Fine at this app's traffic level: the
  translate call only fires after the chat call has already finished, so
  they don't contend for the same request, and model-server sits mostly
  idle between turns at this app's traffic level - but worth knowing if
  you scale up usage.
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
