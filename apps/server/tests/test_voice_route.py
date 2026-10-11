from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import voice


def _client() -> TestClient:
    # A bare app with just this router - app.main's startup hook would
    # otherwise create the real on-disk SQLite file as a side effect.
    app = FastAPI()
    app.include_router(voice.router)
    return TestClient(app)


def test_ping_echoes_each_message():
    with _client().websocket_connect("/voice/ping") as ws:
        ws.send_text("ping 1")
        assert ws.receive_text() == "pong ping 1"
        ws.send_text("ping 2")
        assert ws.receive_text() == "pong ping 2"
