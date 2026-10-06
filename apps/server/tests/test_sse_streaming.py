import time

import pytest

from app.sse_streaming import iter_with_keepalive


def test_forwards_every_item_in_order():
    assert list(iter_with_keepalive(iter(["a", "b", "c"]), interval=5)) == ["a", "b", "c"]


def test_empty_source_yields_nothing():
    assert list(iter_with_keepalive(iter([]), interval=5)) == []


def test_yields_none_keepalives_during_a_gap_longer_than_the_interval():
    # Reported live: a stream showed its first chunk then stopped - no
    # error, just silence - because a reverse proxy's idle-connection
    # timeout closed the connection during a multi-second gap between
    # deltas from a reasoning-style model that can pause mid-generation.
    # Polling the source on a background thread with a short interval
    # means a real gap this long now surfaces as explicit `None` keepalive
    # markers instead of just... nothing, for however long the gap lasts.
    def slow_source():
        yield "first"
        time.sleep(0.12)
        yield "second"

    items = list(iter_with_keepalive(slow_source(), interval=0.03))
    assert items[0] == "first"
    assert items.count(None) >= 2
    assert items[-1] == "second"


def test_reraises_the_sources_own_exception_after_any_items_already_yielded():
    def failing_source():
        yield "ok"
        raise ValueError("boom")

    gen = iter_with_keepalive(failing_source(), interval=5)
    assert next(gen) == "ok"
    with pytest.raises(ValueError, match="boom"):
        next(gen)
