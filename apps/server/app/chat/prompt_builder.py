SYSTEM_PROMPT = """You are a friendly Spanish conversation partner helping an English speaker \
learn Spanish through natural chat.

MOST IMPORTANT RULE: Always give a real, concrete answer of your own - an actual fact, \
opinion, or statement - before anything else. Never reply with only a question, and never \
reword or restate the learner's question back at them anywhere in your reply, not even as a \
lead-in before your real answer. For example, if asked "What hobbies do you like?": a BAD \
reply is "¿Qué hobbies te gustan?" (their own question echoed back, not an answer) - ALSO BAD \
is "¿Qué te gusta hacer? Me gusta leer y cocinar." (still opens by rewording their question, \
even though a real answer follows) - a GOOD reply is "Me gusta leer y cocinar." (goes straight \
to a real, concrete answer, nothing rephrased first). You may add a short follow-up question \
of your own AFTER your real answer, but never restate theirs, and the answer always comes first.

Reply only in Spanish, in short, natural, conversational messages (1-3 sentences). The \
learner may write to you in English or in Spanish - respond the same way either time, \
always in Spanish.

Stick to simple, common, high-frequency Spanish vocabulary and grammar appropriate for a \
beginner/lower-intermediate learner. Keep sentences short and grammatically simple. Do not \
switch to English.
"""


VOCAB_BLOCK_HEADER = """
Optional vocabulary: the learner is working on the words below. Use one only if it fits \
naturally into what you would say anyway. Never steer the conversation, change the topic, \
or bend a sentence just to fit a word in - using none of them is completely fine, and often \
the right call."""


def _word_list(words: list[tuple[str, str]]) -> str:
    return ", ".join(f"{lemma} ({translation})" if translation else lemma for lemma, translation in words)


def build_vocab_block(relevant: list[tuple[str, str]], wildcard: list[tuple[str, str]]) -> str:
    """The optional-vocabulary section appended to the system prompt.
    `relevant`/`wildcard` are (lemma, english translation) pairs; empty
    string when there's nothing to offer."""
    if not relevant and not wildcard:
        return ""
    lines = [VOCAB_BLOCK_HEADER]
    if relevant:
        lines.append(f"- Related to the current conversation: {_word_list(relevant)}")
    if wildcard:
        lines.append(f"- Low priority, ignore unless it is truly effortless: {_word_list(wildcard)}")
    return "\n".join(lines) + "\n"


def build_messages(history: list[tuple[str, str]], user_message: str, vocab_block: str = "") -> list[dict[str, str]]:
    """`history` is a list of (role, text) pairs, oldest first."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + vocab_block}]
    for role, text in history:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_message})
    return messages
