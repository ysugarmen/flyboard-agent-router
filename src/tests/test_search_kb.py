import json
import pytest
import importlib
from pathlib import Path

import src.app.core.config as config

@pytest.fixture()
def search_kb_module(monkeypatch):
    kb_path = Path(__file__).resolve().parent / "test_kb_db" / "test_kb.json"
    monkeypatch.setenv("KB_PATH", str(kb_path))
    monkeypatch.setenv("KB_TOP_K_DEFAULT", "3")

    config.get_settings.cache_clear()

    m = importlib.import_module("src.tools.search_kb")
    m = importlib.reload(m)
    m._KB_INDEX = None
    m._KB_MTIME = None

    return m


def test_search_returns_results_for_basic_query(search_kb_module):
    res = search_kb_module.search_kb("crm")
    assert isinstance(res, dict)
    assert "results" in res
    assert isinstance(res["results"], list)
    assert len(res["results"]) >= 1


def test_search_ranks_expected_top_result_and_normalizes_scores(search_kb_module):
    # With test KB, "crm integration" should strongly match KB-001 vs KB-003.
    res = search_kb_module.search_kb("crm integration", top_k=2)
    results = res["results"]

    assert [r["id"] for r in results] == ["KB-001", "KB-003"]

    # Normalization: top should be 1.0; second should be 0.5 with this KB content
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.5


def test_search_returns_expected_fields(search_kb_module):
    res = search_kb_module.search_kb("crm integration", top_k=1)
    r = res["results"][0]
    assert set(r.keys()) == {"id", "title", "score", "snippet", "tags"}
    assert isinstance(r["tags"], list)
    assert isinstance(r["snippet"], str)


def test_top_k_default_and_bounds(search_kb_module):
    # top_k=None -> uses KB_TOP_K_DEFAULT=3, but only 2 entries match "crm"
    res = search_kb_module.search_kb("crm", top_k=None)
    assert len(res["results"]) == 2

    # top_k larger than cap shouldn't error; results limited by available matches
    res2 = search_kb_module.search_kb("crm", top_k=999)
    assert len(res2["results"]) == 2


def test_invalid_query_raises(search_kb_module):
    with pytest.raises(ValueError):
        search_kb_module.search_kb("")
    with pytest.raises(ValueError):
        search_kb_module.search_kb("   ")
    with pytest.raises(ValueError):
        search_kb_module.search_kb(None)


def test_search_synonym_map(search_kb_module):
    # _SYNONYMS maps 'salesforce' -> 'crm'
    res = search_kb_module.search_kb("salesforce integration", top_k=1)
    assert len(res["results"]) == 1
    assert res["results"][0]["id"] == "KB-001"
    assert res["results"][0]["score"] == 1.0


def test_soft_boost_with_filters(search_kb_module):
    """
    Query 'crm' matches KB-001 (customer) and KB-003 (internal).
    Filters should add bonus for audience=customer and tags=['integration'],
    making KB-001 rank above KB-003.
    """
    res = search_kb_module.search_kb("crm", top_k=2, filters={"audience": "customer", "tags": ["integration"]})
    ids = [r["id"] for r in res["results"]]
    assert ids == ["KB-001", "KB-003"]


def test_troubleshooting_hint_boost(search_kb_module):
    """
    If query tokens include troubleshooting hints (e.g. 'troubleshooting', 'failed'),
    internal audience entries get a +2 bonus. With 'email troubleshooting failed',
    KB-002 should be the top hit.
    """
    res = search_kb_module.search_kb("email troubleshooting failed", top_k=1)
    assert res["results"][0]["id"] == "KB-002"
    assert "SMTP" in res["results"][0]["snippet"]

