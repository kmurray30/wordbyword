"""A short running summary of each session's conversation (table
conversation_summary).

Refreshed by the post-turn background job, after the reply has already
gone out - so the relevance query on the NEXT turn gets the conversation's
gist (topics, situation, what the learner is into) for free, without adding
a serial LLM call in front of the reply. The query still includes the raw
latest messages too; the summary just carries what has scrolled out of them.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.chat.openai_client import chat as model_chat
from app.config import OPENAI_TRANSLATE_MODEL
from app.models import ConversationSummary

# Generous for a 1-3 sentence output: the translate model may be a
# reasoning-style model whose hidden reasoning shares this budget (see
# config.MAX_REPLY_TOKENS's comment).
_SUMMARY_MAX_TOKENS = 800

_SUMMARY_PROMPT = """You maintain a running summary of a casual language-practice chat \
between a learner and a conversation partner. Given the previous summary and the newest \
exchange, write the updated summary: 1-3 short sentences in English covering the topics \
being discussed, the situation or setting, and anything the learner has said about \
themselves or their interests. Favor what is current; drop topics the conversation has \
clearly moved on from. Output only the summary text."""


def get_summary(session: Session, session_id: str) -> str:
    row = session.get(ConversationSummary, session_id)
    return row.summary if row else ""


def summarize(previous: str, user_message: str, assistant_reply: str) -> str:
    """One LLM call; raises ModelServerUnavailableError on failure."""
    exchange = f"Learner: {user_message}\nPartner: {assistant_reply}"
    messages = [
        {"role": "system", "content": _SUMMARY_PROMPT},
        {"role": "user", "content": f"Previous summary:\n{previous or '(none yet)'}\n\nNewest exchange:\n{exchange}"},
    ]
    return model_chat(messages, max_tokens=_SUMMARY_MAX_TOKENS, model=OPENAI_TRANSLATE_MODEL).strip()


def update_summary(session: Session, session_id: str, user_message: str, assistant_reply: str) -> str:
    updated = summarize(get_summary(session, session_id), user_message, assistant_reply)
    if not updated:
        return get_summary(session, session_id)
    row = session.get(ConversationSummary, session_id)
    if row is None:
        row = ConversationSummary(session_id=session_id)
        session.add(row)
    row.summary = updated
    row.updated_at = datetime.utcnow()
    session.commit()
    return updated


def clear_summary(session: Session, session_id: str) -> None:
    row = session.get(ConversationSummary, session_id)
    if row is not None:
        session.delete(row)
