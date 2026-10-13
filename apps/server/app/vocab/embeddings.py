"""Text embeddings via DeepInfra's OpenAI-compatible /embeddings endpoint,
plus the word embedding cache (table word_embedding).

No vector database: the candidate words are at most a few thousand words, so ranking it
is a single matrix-vector product over an in-memory numpy array - well
under a millisecond.
"""

import logging

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEEPINFRA_API_TOKEN, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.models import WordEmbedding

DEEPINFRA_EMBEDDINGS_URL = "https://api.deepinfra.com/v1/openai/embeddings"

# Qwen3-Embedding is instruction-aware on the query side only: documents
# (the words) are embedded bare, queries carry a one-line task description.
_QUERY_INSTRUCTION = (
    "Given a snippet of a {language}-learning conversation, retrieve {language} vocabulary words "
    "that would come up naturally in the next reply"
)

_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    pass


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(texts: list[str], timeout: float = 15.0) -> np.ndarray:
    """Returns one L2-normalized float32 row per input text, truncated to
    EMBEDDING_DIMENSIONS (renormalized after truncation, which is what makes
    a Matryoshka prefix usable for cosine similarity)."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
    if not DEEPINFRA_API_TOKEN:
        raise EmbeddingUnavailableError("DEEPINFRA_API_TOKEN is not configured on the server")

    headers = {"Authorization": f"Bearer {DEEPINFRA_API_TOKEN}"}
    rows: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        payload = {"model": EMBEDDING_MODEL, "input": batch, "encoding_format": "float"}
        # One retry on a transport-level failure (a stalled connection often
        # clears on a fresh one - same lesson as tts/deepinfra_client.py);
        # an HTTP error status means DeepInfra actually answered, so no retry.
        last_error: httpx.HTTPError | None = None
        for _ in range(2):
            try:
                response = httpx.post(DEEPINFRA_EMBEDDINGS_URL, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                raise EmbeddingUnavailableError(
                    f"DeepInfra embeddings request failed ({exc.response.status_code}): {exc.response.text}"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
        else:
            raise EmbeddingUnavailableError(f"Could not reach DeepInfra embeddings: {last_error!r}") from last_error

        data = sorted(response.json().get("data") or [], key=lambda item: item.get("index", 0))
        if len(data) != len(batch):
            raise EmbeddingUnavailableError(f"DeepInfra returned {len(data)} embeddings for {len(batch)} inputs")
        rows.extend(item["embedding"][:EMBEDDING_DIMENSIONS] for item in data)

    return _normalize(np.asarray(rows, dtype=np.float32))


def embed_query(text: str, language_name: str = "Spanish") -> np.ndarray:
    instruction = _QUERY_INSTRUCTION.format(language=language_name)
    return embed_texts([f"Instruct: {instruction}\nQuery:{text}"])[0]


def word_document(lemma: str, translation: str) -> str:
    """What gets embedded for one word: the lemma plus its English gloss when
    known, so the vector carries the word's meaning in both languages the
    conversation might be in."""
    return f"{lemma}: {translation}" if translation else lemma


# (language, lemma) -> vector for the current EMBEDDING_MODEL. Rows are
# immutable once written (a re-embed only ever happens under a different
# model name), so the cache only ever needs to learn about new words, never
# invalidate.
_cache: dict[tuple[str, str], np.ndarray] = {}


def load_vectors(session: Session, language: str, lemmas: list[str]) -> dict[str, np.ndarray]:
    """Cached vectors for whichever of `lemmas` have been embedded so far."""
    missing = [lemma for lemma in lemmas if (language, lemma) not in _cache]
    if missing:
        rows = session.execute(
            select(WordEmbedding.lemma, WordEmbedding.vector).where(
                WordEmbedding.model == EMBEDDING_MODEL,
                WordEmbedding.language == language,
                WordEmbedding.lemma.in_(missing),
            )
        ).all()
        for lemma, blob in rows:
            _cache[(language, lemma)] = np.frombuffer(blob, dtype=np.float32)
    return {lemma: _cache[(language, lemma)] for lemma in lemmas if (language, lemma) in _cache}


def remember_vectors(language: str, words: list[tuple[str, str]]) -> None:
    """Embeds `words` ((lemma, translation) pairs) into the in-memory cache
    only - no DB write, so it's safe on the reply's read-only request path.
    embed_missing later persists them without paying for the API call again."""
    vectors = embed_texts([word_document(lemma, translation) for lemma, translation in words])
    for (lemma, _), vector in zip(words, vectors):
        _cache[(language, lemma)] = vector


def embed_missing(session: Session, language: str, words: list[tuple[str, str]], limit: int = 256) -> int:
    """Embeds up to `limit` of `words` ((lemma, translation) pairs) that
    don't have a vector under the current model yet, and commits them.
    Returns how many it added. Run from the post-turn background job (and
    when the word list changes), never on the reply's request path."""
    if not words:
        return 0
    have = set(
        session.scalars(
            select(WordEmbedding.lemma).where(
                WordEmbedding.model == EMBEDDING_MODEL,
                WordEmbedding.language == language,
                WordEmbedding.lemma.in_([lemma for lemma, _ in words]),
            )
        ).all()
    )
    todo = [(lemma, translation) for lemma, translation in words if lemma not in have][:limit]
    if not todo:
        return 0

    cached = [(lemma, _cache[(language, lemma)]) for lemma, _ in todo if (language, lemma) in _cache]
    fresh = [(lemma, translation) for lemma, translation in todo if (language, lemma) not in _cache]
    fetched = embed_texts([word_document(lemma, translation) for lemma, translation in fresh]) if fresh else []
    for (lemma, vector) in cached + [(lemma, v) for (lemma, _), v in zip(fresh, fetched)]:
        session.add(
            WordEmbedding(
                language=language, lemma=lemma, model=EMBEDDING_MODEL, vector=vector.astype(np.float32).tobytes()
            )
        )
    session.commit()
    return len(todo)
