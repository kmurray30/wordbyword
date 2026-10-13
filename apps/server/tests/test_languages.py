"""Everything that varies by the language being learned: the registry,
tokenizing, readings, gloss parsing and speech-to-text settings."""

from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.glosses import build_glosses, complete_phrase_end, reply_tokens
from app.chat.prompt_builder import build_messages
from app.db import Base
from app.languages import get_language, has_foreign_script, language_name, with_script
from app.models import ChatMessage
from app.routes.chat import message_glosses, take_turn
from app.routes.translate import readings, tag_input
from app.schemas import ChatTurnRequest, MessageGlosses, MessageGlossesRequest, ReadingsRequest, TagInputRequest, WordGloss
from app.stt.deepgram_client import build_listen_url, unspace
from app.translate.lemmatizer import analyze, looks_like_target
from app.translate.llm_translate import _extract_gloss_json, gloss_message, translate_text
from app.translate.readings import readings_for, romanize_korean


def _session():
    engine = create_engine("sqlite:///:memory:")
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


# --- registry -----------------------------------------------------------------


def test_unknown_or_missing_code_falls_back_to_spanish():
    assert get_language(None).code == "es"
    assert get_language("xx").code == "es"
    assert get_language("zh-Hant").code == "zh"


def test_mandarin_script_variant_is_in_the_prompt_name():
    assert with_script("zh", "traditional") == "zh-Hant"
    assert with_script("zh", None) == "zh-Hans"
    assert with_script("ja", "traditional") == "ja"
    assert "Traditional" in language_name("zh-Hant")
    assert language_name("ko") == "Korean"


def test_foreign_script_depends_on_the_language():
    assert has_foreign_script("Hola 你好", "es")
    assert not has_foreign_script("こんにちは、元気？", "ja")
    assert has_foreign_script("안녕 こんにちは", "zh")
    assert not has_foreign_script("안녕하세요 hello", "ko")


def test_partner_prompt_is_in_the_language_being_learned():
    system = build_messages([], "hi", "ja")[0]["content"]
    assert "Japanese" in system and "Spanish" not in system
    assert "Traditional" in build_messages([], "hi", "zh-Hant")[0]["content"]


# --- tokenizing ---------------------------------------------------------------


def test_japanese_splits_by_script_and_counts_as_target():
    tokens = analyze("私はcoffeeが好き。", "ja")
    surfaces = [t.surface for t in tokens]
    assert surfaces == ["私", "は", "coffee", "が", "好", "き", "。"]
    assert {t.surface: t.is_target for t in tokens}["coffee"] is False
    assert {t.surface: t.is_target for t in tokens}["私"] is True


def test_french_words_are_told_from_english():
    by_surface = {t.surface: t.is_target for t in analyze("Je veux un café with milk", "fr")}
    assert by_surface["veux"] and by_surface["café"]
    assert not by_surface["with"]
    assert looks_like_target("Je suis très content", "fr")
    assert not looks_like_target("I am very happy", "fr")


def test_korean_and_chinese_majority_vote():
    assert looks_like_target("저는 학생이에요", "ko")
    assert not looks_like_target("I am a student", "ko")
    assert looks_like_target("我是学生", "zh")


def test_tag_input_keeps_single_character_words_in_chinese():
    response = tag_input(TagInputRequest(text="我想喝coffee", language="zh"))
    assert [(t.surface, t.is_target) for t in response.tokens] == [("我想喝", True), ("coffee", False)]


# --- glosses ------------------------------------------------------------------


def test_cjk_phrases_survive_without_spaces():
    reply = '{"words": [], "phrases": [{"surface": "加油", "gloss": "go for it"}, {"surface": "我", "gloss": "I"}]}'
    _translation, _words, phrases = _extract_gloss_json(reply, spaced=False)
    assert [p["surface"] for p in phrases] == ["加油"]


def test_gloss_prompt_has_the_languages_own_examples():
    with patch("app.translate.llm_translate.model_chat", return_value='{"translation": "", "words": []}') as mock:
        gloss_message("日本語", "en", "ja")
    system = mock.call_args.args[0][0]["content"]
    assert "Japanese" in system and "particle" in system
    assert "seguir" not in system


def test_a_chinese_reply_is_not_garbled_in_a_chinese_lesson():
    with patch("app.translate.llm_translate.model_chat", return_value="你好") as mock:
        assert translate_text("hello", "en", "zh") == "你好"
    assert mock.call_count == 1


def test_french_phrase_ending_on_an_article_is_completed():
    text = "On va faire la fête ce soir"
    start = text.index("faire")
    end = start + len("faire la")
    assert text[start : complete_phrase_end(text, start, end, "fr")] == "faire la fête"


def test_phrase_completion_does_nothing_for_japanese():
    text = "気をつけて ください"
    assert complete_phrase_end(text, 0, 2, "ja") == 2


def test_reply_tokens_come_from_the_glossed_words_outside_spanish():
    text = "私は学生です。"
    glosses = build_glosses(
        text,
        [
            {"surface": "私", "gloss": "I"},
            {"surface": "は", "gloss": "(topic)"},
            {"surface": "学生", "gloss": "student"},
            {"surface": "です", "gloss": "am"},
        ],
        [],
        "ja",
    )
    tokens = reply_tokens(text, "ja", glosses)
    assert [t.surface for t in tokens] == ["私", "は", "学生", "です", "。"]
    assert all(text[t.start : t.end] == t.surface for t in tokens)
    # No glosses: the plain tokenizer stands in.
    assert [t.surface for t in reply_tokens("学生です", "ja", MessageGlosses())] == ["学生", "です"]


def test_reply_tokens_skip_overlapping_words():
    text = "学生です"
    glosses = MessageGlosses(words=[WordGloss(start=0, end=2, gloss="a"), WordGloss(start=1, end=3, gloss="b")])
    assert [t.surface for t in reply_tokens(text, "ja", glosses)] == ["学生", "です"]


# --- chat in another language ---------------------------------------------------


def test_a_turn_in_japanese_is_saved_as_japanese_with_word_bank_entries_per_word():
    session = _session()
    reply = "私は学生です。"
    gloss_json = (
        '{"translation": "I am a student.", "words": [{"surface": "私", "gloss": "I"}, '
        '{"surface": "は", "gloss": "(topic)"}, {"surface": "学生", "gloss": "student"}, '
        '{"surface": "です", "gloss": "am"}], "phrases": []}'
    )
    with (
        patch("app.routes.chat.model_chat", return_value=reply) as chat,
        patch("app.translate.llm_translate.model_chat", return_value=gloss_json),
    ):
        response = take_turn(ChatTurnRequest(message="hello", session_id="s-ja", language="ja"), session)

    assert "Japanese" in chat.call_args.args[0][0]["content"]
    assert [t.surface for t in response.tokens] == ["私", "は", "学生", "です", "。"]
    assert response.translation == "I am a student."
    rows = session.query(ChatMessage).all()
    assert {r.language for r in rows} == {"ja"}
    from app.models import WordBankEntry

    assert {(e.language, e.lemma) for e in session.query(WordBankEntry).all()} == {
        ("ja", "私"),
        ("ja", "は"),
        ("ja", "学生"),
        ("ja", "です"),
    }


def test_message_glosses_use_the_messages_own_language():
    session = _session()
    row = ChatMessage(session_id="s", language="ko", role="assistant", text="안녕하세요")
    session.add(row)
    session.commit()
    with patch(
        "app.routes.chat.llm_translate.gloss_message", return_value=("", [{"surface": "안녕하세요", "gloss": "hello"}], [])
    ) as gloss:
        message_glosses(MessageGlossesRequest(message_id=row.id), session)
    assert gloss.call_args.args[2] == "ko"


# --- readings -----------------------------------------------------------------


def test_furigana_goes_over_the_kanji_only():
    _text, found = readings_for("東京へ行きます", "ja")
    pieces = [("東京へ行きます"[r.start : r.end], r.kana, r.roman) for r in found]
    assert ("東京", "とうきょう", "toukyou") in pieces
    assert ("へ", "", "e") in pieces  # the particle, read "e"
    assert ("行", "い", "i") in pieces  # okurigana き stays outside the furigana
    assert ("き", "", "ki") in pieces
    assert [r.cont for r in found if "東京へ行きます"[r.start : r.end] in ("行", "き")] == [False, True]


def test_pinyin_reads_characters_in_context_and_traditional_converts():
    shown, found = readings_for("银行行长", "zh-Hant")
    assert shown == "銀行行長"
    assert [r.roman for r in found] == ["yín", "háng", "háng", "zhǎng"]
    shown, _ = readings_for("銀行", "zh")
    assert shown == "银行"


def test_korean_romanization_follows_sound_changes():
    assert romanize_korean("안녕하세요") == "annyeonghaseyo"
    assert romanize_korean("감사합니다") == "gamsahamnida"
    assert romanize_korean("한국어") == "hangugeo"
    assert romanize_korean("설날") == "seollal"


def test_latin_languages_have_no_readings():
    assert readings_for("Bonjour", "fr") == ("Bonjour", [])


def test_readings_route_returns_each_text():
    response = readings(ReadingsRequest(texts=["中文", "hi"], language="zh"))
    assert [item.text for item in response.items] == ["中文", "hi"]
    assert [r.roman for r in response.items[0].readings] == ["zhōng", "wén"]
    assert response.items[1].readings == []


# --- speech to text -------------------------------------------------------------


def test_listen_url_per_language():
    assert "language=multi" in build_listen_url([], "fr")
    korean = build_listen_url(["학생"], "ko")
    assert "language=ko" in korean and "keyterm" in korean
    chinese = build_listen_url(["学生"], "zh-Hant")
    assert "language=zh-TW" in chinese and "model=nova-2" in chinese
    assert "keyterm" not in chinese  # a Nova-3 feature


def test_spaces_between_chinese_words_are_dropped():
    assert unspace("我 喜欢 中文", "zh") == "我喜欢中文"
    assert unspace("hola amigo", "es") == "hola amigo"
