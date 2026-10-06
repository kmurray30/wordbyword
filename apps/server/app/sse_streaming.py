"""Keepalive wrapper shared by every streamed (SSE) route - /chat/turn/stream
and /translate/coach alike.

Reported live: a stream showed its first chunk, then nothing - no error,
just silence - until reloading the page revealed the model HAD finished
generating (and the server had persisted the full reply) the whole time.
Root cause: OPENAI_CHAT_MODEL is a reasoning-style model (see config.py's
own comments on it) that can pause for several seconds between visible
output tokens while it "thinks" - a gap plain or edge-proxy idle-connection
timeouts (commonly 30-60s, but can be shorter) read as a dead connection
and silently close, well before this server's own httpx read timeout (60s)
would ever notice anything wrong. The server-side generator, oblivious to
that closed connection (nothing here checks for client disconnection),
keeps running to completion and persists the reply regardless - which is
exactly why a reload shows the full message: it was never actually lost,
just no longer being delivered to that now-dead connection.

Fix: never let the underlying iterator's own pace decide how long the SSE
response goes quiet. Running it on a background thread and polling it with
a timeout lets this yield a cheap keepalive comment line whenever nothing
real has arrived in a while, keeping the connection visibly alive to every
hop in between regardless of how long the model itself pauses.
"""

import queue
import threading
from collections.abc import Iterator

# Comfortably under the 30-60s idle timeouts typical of a reverse proxy
# (Railway's edge included) - frequent enough that no such timeout ever
# gets the chance to fire, cheap enough (one short comment line) that it
# costs nothing to send this often.
KEEPALIVE_INTERVAL_SECONDS = 12.0

# Any line starting with ":" is an SSE comment - part of the spec, ignored
# by every conforming client (confirmed against this app's own
# readServerSentEvents, which only ever looks for "event:"/"data:" lines).
KEEPALIVE_LINE = ": keepalive\n\n"


def _drain_to_queue(source: Iterator, sink: "queue.Queue") -> None:
    try:
        for item in source:
            sink.put(("item", item))
    except Exception as exc:  # noqa: BLE001 - re-raised as-is on the caller's own thread below
        sink.put(("error", exc))
    finally:
        sink.put(("done", None))


def iter_with_keepalive(source: Iterator, interval: float = KEEPALIVE_INTERVAL_SECONDS) -> Iterator:
    """Wraps `source` (any iterator/generator, run on a background thread so
    it can be polled with a timeout) and yields each of its items as soon
    as it's ready, interleaved with `None` whenever nothing has arrived
    within `interval` seconds - the caller yields a keepalive comment line
    for each `None` and simply continues, never seeing a gap longer than
    `interval` regardless of how long `source` itself goes quiet between
    items. An exception raised while draining `source` is re-raised here,
    on the caller's own thread, once it's actually reached in order - same
    as iterating `source` directly would, just without the long silent gap
    that direct iteration risked."""
    q: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_drain_to_queue, args=(source, q), daemon=True)
    thread.start()
    while True:
        try:
            kind, payload = q.get(timeout=interval)
        except queue.Empty:
            yield None
            continue
        if kind == "item":
            yield payload
        elif kind == "error":
            raise payload
        else:  # "done"
            return
