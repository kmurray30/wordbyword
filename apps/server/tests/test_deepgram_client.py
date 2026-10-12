import asyncio
import json
from unittest.mock import patch

import pytest

from app.stt.deepgram_client import (
    MAX_KEYTERMS,
    STTUnavailableError,
    build_listen_url,
    normalize_message,
    stream_transcripts,
)


def test_listen_url_requests_code_switching_and_raw_pcm():
    url = build_listen_url()
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    assert "language=multi" in url
    assert "encoding=linear16" in url
    assert "sample_rate=16000" in url
    assert "interim_results=true" in url
    assert "utterance_end_ms=1200" in url
    assert "vad_events=true" in url


def test_listen_url_repeats_keyterm_per_word_and_caps_the_count():
    url = build_listen_url(["correr", "playa"])
    assert "keyterm=correr" in url
    assert "keyterm=playa" in url
    too_many = [f"w{i}" for i in range(MAX_KEYTERMS + 20)]
    assert build_listen_url(too_many).count("keyterm=") == MAX_KEYTERMS


def test_normalize_interim_and_final_results():
    raw = {
        "type": "Results",
        "is_final": False,
        "channel": {
            "alternatives": [
                {
                    "transcript": "te gusta running",
                    "words": [
                        {"word": "te", "punctuated_word": "Te", "confidence": 0.98},
                        {"word": "gusta", "confidence": 0.61},
                        {"word": "running", "confidence": 0.9},
                    ],
                }
            ]
        },
    }
    event = normalize_message(raw)
    assert event is not None
    assert event.kind == "interim"
    assert event.text == "te gusta running"
    assert event.words == [
        {"word": "Te", "confidence": 0.98},
        {"word": "gusta", "confidence": 0.61},
        {"word": "running", "confidence": 0.9},
    ]

    raw["is_final"] = True
    assert normalize_message(raw).kind == "final"


def test_normalize_drops_empty_results_and_metadata():
    assert normalize_message({"type": "Metadata", "request_id": "x"}) is None
    assert normalize_message({"type": "Results", "channel": {"alternatives": [{"transcript": "   "}]}}) is None
    assert normalize_message({"type": "Results", "channel": {"alternatives": []}}) is None


def test_normalize_control_events():
    assert normalize_message({"type": "UtteranceEnd", "last_word_end": 3.2}).kind == "utterance_end"
    assert normalize_message({"type": "SpeechStarted"}).kind == "speech_started"


async def _no_frames():
    return
    yield  # pragma: no cover - makes this an async generator


def test_stream_raises_when_key_missing():
    with patch("app.stt.deepgram_client.DEEPGRAM_API_KEY", ""):
        with pytest.raises(STTUnavailableError, match="not configured"):

            async def run():
                async for _ in stream_transcripts(_no_frames()):
                    pass

            asyncio.run(run())


def test_stream_wraps_connection_failure():
    async def failing_connect(*args, **kwargs):
        raise OSError("boom")

    with patch("app.stt.deepgram_client.DEEPGRAM_API_KEY", "dg-fake"):
        with patch("app.stt.deepgram_client.websockets.connect", side_effect=failing_connect):
            with pytest.raises(STTUnavailableError, match="Could not connect"):

                async def run():
                    async for _ in stream_transcripts(_no_frames()):
                        pass

                asyncio.run(run())


class _FakeDeepgramSocket:
    """Stands in for the websockets connection: records everything sent,
    and replays canned server messages once the first audio frame arrives."""

    def __init__(self, replies: list[str]):
        self.sent: list = []
        self.replies = replies
        self.closed = False
        self._got_audio = asyncio.Event()

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, (bytes, bytearray)):
            self._got_audio.set()

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        await self._got_audio.wait()
        for reply in self.replies:
            yield reply
        # Stay open (like the real socket would) until the caller is done.
        await asyncio.sleep(0.6)


def test_stream_keeps_pinging_through_a_long_gap_and_still_forwards_later_audio():
    # Regression for the live "Voice connection closed" failure: during a
    # muted gap no frames arrive; a KeepAlive must still go out on cadence
    # AND audio arriving after the gap must still be forwarded (the old
    # wait_for-based gap detection cancelled the frame iterator and killed
    # the sender on the first gap).
    fake = _FakeDeepgramSocket([json.dumps({"type": "SpeechStarted"})])

    async def connect(*args, **kwargs):
        return fake

    async def frames():
        yield b"\x01\x02"
        await asyncio.sleep(0.35)  # a gap several keepalive intervals long
        yield b"\x03\x04"
        yield None

    async def run():
        events = []
        async for event in stream_transcripts(frames()):
            events.append(event)
        return events

    with (
        patch("app.stt.deepgram_client.DEEPGRAM_API_KEY", "dg-fake"),
        patch("app.stt.deepgram_client._KEEPALIVE_INTERVAL_S", 0.1),
        patch("app.stt.deepgram_client.websockets.connect", side_effect=connect),
    ):
        events = asyncio.run(run())

    assert [e.kind for e in events] == ["speech_started"]
    audio = [s for s in fake.sent if isinstance(s, bytes)]
    text = [json.loads(s) for s in fake.sent if isinstance(s, str)]
    assert audio == [b"\x01\x02", b"\x03\x04"]  # audio after the gap still forwarded
    assert sum(1 for m in text if m["type"] == "KeepAlive") >= 2
    assert text[-1] == {"type": "CloseStream"}
    assert fake.closed
