import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agent import runner


class DummyResponses:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **_kwargs):
        return self._responses.pop(0)


class DummyOpenAI:
    def __init__(self, api_key, responses):
        self.responses = DummyResponses(responses)


def test_run_task_orchestrates_tool_loop():
    tool_call = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="search_kb",
        arguments=json.dumps({"query": "crm"}),
    )
    message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(text="Here is the answer.")],
    )
    responses = [SimpleNamespace(output=[tool_call]), SimpleNamespace(output=[message])]

    def dummy_openai(api_key):
        return DummyOpenAI(api_key=api_key, responses=responses)

    original_api_key = runner.settings.OPENAI_API_KEY
    try:
        runner.settings.OPENAI_API_KEY = "test-key"
        with patch("src.agent.runner.OpenAI", side_effect=dummy_openai), patch(
            "src.agent.runner.search_kb", return_value={"results": [{"id": "KB-001"}]}
        ) as mock_search:
            result = runner.run_task("Find CRM guidance")

        assert result["final_answer"] == "Here is the answer."
        assert result["metrics"]["openai_calls"] == 2
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search_kb"
        assert result["tool_calls"][0]["result"] == mock_search.return_value
    finally:
        runner.settings.OPENAI_API_KEY = original_api_key
