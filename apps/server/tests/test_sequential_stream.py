from app.translate.sequential_stream import Field, stream_sequential_fields


def _drain(chunks, fields, initial_buffer=""):
    gen = stream_sequential_fields(iter(chunks), fields, initial_buffer)
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


def test_single_field_streams_then_completes():
    events, leftover = _drain(["hola", " mundo", "<<<END>>>"], [Field("x", "<<<END>>>")])
    # The withheld tail (len(marker)-1 characters) naturally splits delivery
    # into more than one "x_chunk" as text safely clears that tail across
    # chunk boundaries - that's correct incremental streaming, not a bug;
    # what matters is the deltas concatenate back to the full text and the
    # final "x_complete" carries the whole thing.
    chunk_events = [e for e in events if e[0] == "x_chunk"]
    assert "".join(e[1] for e in chunk_events) == "hola mundo"
    assert events[-1] == ("x_complete", "hola mundo")
    assert leftover == ""


def test_multiple_fields_in_order():
    events, leftover = _drain(
        ["a<<<A>>>", "b<<<B>>>", "c<<<C>>>"],
        [Field("f1", "<<<A>>>"), Field("f2", "<<<B>>>"), Field("f3", "<<<C>>>")],
    )
    kinds = [e[0] for e in events]
    assert kinds == ["f1_chunk", "f1_complete", "f2_chunk", "f2_complete", "f3_chunk", "f3_complete"]
    assert [e[1] for e in events if e[0].endswith("_complete")] == ["a", "b", "c"]
    assert leftover == ""


def test_marker_split_across_chunks_never_leaks_a_fragment():
    # The marker can land anywhere relative to chunk boundaries - split
    # right in the middle of it, repeatedly, across many tiny chunks.
    full = "hello world<<<END>>>"
    chunks = [full[i : i + 3] for i in range(0, len(full), 3)]
    events, _ = _drain(chunks, [Field("x", "<<<END>>>")])
    chunk_text = "".join(e[1] for e in events if e[0] == "x_chunk")
    assert chunk_text == "hello world"
    assert "<" not in chunk_text


def test_handles_fields_whose_markers_have_different_lengths():
    # The withheld tail is sized once, up front, to the LONGEST marker
    # across every field (not recomputed per field) - a deliberately more
    # conservative choice than strictly necessary (only the CURRENT
    # field's own marker can ever match mid-scan), simple enough to not be
    # worth optimizing away. This just confirms mixing short and long
    # markers in one `fields` list still streams/completes correctly.
    short_marker = "<<<A>>>"
    long_marker = "<<<MUCH_LONGER_MARKER>>>"
    chunks = ["text" + short_marker, "more<<<MUCH_LONGER_MARKER>>>"]
    events, _ = _drain(chunks, [Field("f1", short_marker), Field("f2", long_marker)])
    assert events[0] == ("f1_chunk", "text")
    assert events[1] == ("f1_complete", "text")
    assert events[2] == ("f2_chunk", "more")
    assert events[3] == ("f2_complete", "more")


def test_extra_metadata_is_echoed_back():
    events, _ = _drain(["spain<<<ES>>>"], [Field("opt", "<<<ES>>>", extra=(2, "spanish"))])
    assert events == [
        ("opt_chunk", 2, "spanish", "spain"),
        ("opt_complete", 2, "spanish", "spain"),
    ]


def test_field_never_completes_when_chunks_exhaust_first():
    events, leftover = _drain(["partial text, no marker"], [Field("x", "<<<END>>>")])
    kinds = [e[0] for e in events]
    assert kinds == ["x_chunk"]
    assert leftover == "partial text, no marker"


def test_later_field_incomplete_leaves_earlier_fields_fully_usable():
    events, leftover = _drain(
        ["a<<<A>>>", "incomplete b, no marker"],
        [Field("f1", "<<<A>>>"), Field("f2", "<<<B>>>")],
    )
    kinds = [e[0] for e in events]
    assert kinds == ["f1_chunk", "f1_complete", "f2_chunk"]
    assert leftover == "incomplete b, no marker"


def test_resumes_from_a_previous_calls_leftover_buffer_via_initial_buffer():
    # Mirrors coach_draft_stream's own two-call pattern: parse one leading
    # field first, inspect its value, THEN decide the rest of the fields -
    # continuing against the SAME underlying chunk iterator, seeded with
    # whatever text the first call had already pulled but not consumed.
    chunk_iter = iter(["VERDICT_VALUE<<<V>>>rest of the meaning<<<M>>>"])
    events1, leftover1 = _drain_iter(chunk_iter, [Field("verdict", "<<<V>>>")])
    assert events1 == [("verdict_chunk", "VERDICT_VALUE"), ("verdict_complete", "VERDICT_VALUE")]
    events2, leftover2 = _drain_iter(chunk_iter, [Field("meaning", "<<<M>>>")], initial_buffer=leftover1)
    assert events2 == [("meaning_chunk", "rest of the meaning"), ("meaning_complete", "rest of the meaning")]
    assert leftover2 == ""


def _drain_iter(chunk_iter, fields, initial_buffer=""):
    gen = stream_sequential_fields(chunk_iter, fields, initial_buffer)
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


def test_empty_fields_list_returns_initial_buffer_untouched():
    events, leftover = _drain(["whatever"], [], initial_buffer="pre-existing")
    assert events == []
    assert leftover == "pre-existing"


def test_initial_buffer_already_containing_a_complete_field_is_drained_before_pulling_new_chunks():
    events, leftover = _drain(["more"], [Field("x", "<<<END>>>")], initial_buffer="already here<<<END>>>")
    assert events == [("x_chunk", "already here"), ("x_complete", "already here")]
    # The extra "more" chunk is never pulled since the single field already
    # completed entirely from initial_buffer - leftover is empty, not "more".
    assert leftover == ""
