from src.tools import search_kb as search_module


def test_search_kb_returns_top_match(kb_json):
    original_db_path = search_module._DB_PATH
    original_index = search_module._KB_INDEX
    original_mtime = search_module._KB_MTIME
    try:
        search_module._DB_PATH = kb_json
        search_module._KB_INDEX = None
        search_module._KB_MTIME = None

        result = search_module.search_kb(
            "crm integration", top_k=2, filters={"audience": "customer"}
        )

        assert len(result["results"]) >= 1
        assert result["results"][0]["id"] == "KB-001"
        assert "crm" in result["results"][0]["snippet"].lower()
    finally:
        search_module._DB_PATH = original_db_path
        search_module._KB_INDEX = original_index
        search_module._KB_MTIME = original_mtime
