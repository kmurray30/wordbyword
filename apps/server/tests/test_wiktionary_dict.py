import json

import pytest

from app.translate import wiktionary_dict


@pytest.fixture(autouse=True)
def _clear_cache():
    wiktionary_dict._data.cache_clear()
    yield
    wiktionary_dict._data.cache_clear()


def test_lookup_returns_empty_when_data_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wiktionary_dict, "DATA_PATH", tmp_path / "missing.json")
    assert wiktionary_dict.lookup_es_to_en("banco") == []
    assert wiktionary_dict.lookup_en_to_es("bro") == []


def test_lookup_reads_bundled_data_file(tmp_path, monkeypatch):
    data_path = tmp_path / "dict.json"
    data_path.write_text(
        json.dumps(
            {
                "es_to_en": {"banco": [{"translation": "bank", "description": "financial institution"}]},
                "en_to_es": {"bro": [{"translation": "tío", "description": "(informal) close friend"}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wiktionary_dict, "DATA_PATH", data_path)

    assert wiktionary_dict.lookup_es_to_en("BANCO") == [{"translation": "bank", "description": "financial institution"}]
    assert wiktionary_dict.lookup_en_to_es("bro") == [{"translation": "tío", "description": "(informal) close friend"}]
    assert wiktionary_dict.lookup_es_to_en("nope") == []
