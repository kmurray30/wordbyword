SYSTEM_PROMPT_TEMPLATE = """You are a friendly Spanish conversation partner helping an English speaker \
learn Spanish through natural chat. Reply only in Spanish, in short, natural, \
conversational messages (1-3 sentences).

The learner may write to you in English or in Spanish - respond the same way either \
time: understand what they mean and reply naturally in Spanish, as a real conversation \
partner would. Do NOT just translate their message back to them, and do not repeat \
their question at them - actually answer it, or react to it, the way a person would.

Vocabulary rules, most important first:
1. Prefer using these words the learner is reviewing, if they fit naturally: {reinforce}
2. You may introduce a few of these new words if it fits naturally, but don't force all of them in: {new_words}
3. Otherwise, stick to simple, common, high-frequency Spanish vocabulary and grammar \
appropriate for a beginner/lower-intermediate learner.
4. Keep sentences short and grammatically simple. Do not switch to English.
"""


def build_system_prompt(reinforce_lemmas: list[str], new_lemmas: list[str]) -> str:
    reinforce = ", ".join(reinforce_lemmas) if reinforce_lemmas else "(none yet - this is early on)"
    new_words = ", ".join(new_lemmas) if new_lemmas else "(none)"
    return SYSTEM_PROMPT_TEMPLATE.format(reinforce=reinforce, new_words=new_words)


def build_messages(
    history: list[tuple[str, str]],
    reinforce_lemmas: list[str],
    new_lemmas: list[str],
    user_message: str,
) -> list[dict[str, str]]:
    """`history` is a list of (role, text) pairs, oldest first."""
    messages = [{"role": "system", "content": build_system_prompt(reinforce_lemmas, new_lemmas)}]
    for role, text in history:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_message})
    return messages
