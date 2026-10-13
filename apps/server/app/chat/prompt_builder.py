from app.languages import get_language, language_name


def system_prompt(language: str) -> str:
    """The conversation partner's instructions, for the language being
    learned (`language` may be a script variant like "zh-Hant")."""
    name = language_name(language)
    style = get_language(language).partner_style
    return f"""You are a friendly {name} conversation partner helping an English speaker \
learn {name} through natural chat.

MOST IMPORTANT RULE: Always give a real, concrete answer of your own - an actual fact, \
opinion, or statement - before anything else. Never reply with only a question, and never \
reword or restate the learner's question back at them anywhere in your reply, not even as a \
lead-in before your real answer. For example, if asked "What hobbies do you like?": a BAD \
reply asks them "What hobbies do you like?" back in {name} (their own question echoed back, \
not an answer) - ALSO BAD is opening with "What do you like to do?" and then answering (still \
opens by rewording their question, even though a real answer follows) - a GOOD reply is "I \
like reading and cooking." in {name} (goes straight to a real, concrete answer, nothing \
rephrased first). You may add a short follow-up question of your own AFTER your real answer, \
but never restate theirs, and the answer always comes first.

Reply only in {name}, in short, natural, conversational messages (1-3 sentences). The \
learner may write to you in English or in {name} (possibly romanized, if they can't type its \
script yet) - respond the same way either time, always in {name}.

Stick to simple, common, high-frequency {name} vocabulary and grammar appropriate for a \
beginner/lower-intermediate learner. Keep sentences short and grammatically simple. Do not \
switch to English. {style}
""".rstrip() + "\n"


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


def build_messages(
    history: list[tuple[str, str]], user_message: str, language: str = "es", vocab_block: str = ""
) -> list[dict[str, str]]:
    """`history` is a list of (role, text) pairs, oldest first."""
    messages = [{"role": "system", "content": system_prompt(language) + vocab_block}]
    for role, text in history:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_message})
    return messages
