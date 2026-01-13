import json
import sys
import importlib
from dataclasses import dataclass
from typing import List, Any
from types import SimpleNamespace

import pytest


# -----------------------
# Mock OpenAI Responses API objects
# -----------------------
def _msg(text: str):
    # Matches what _extract_text_from_response() expects
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(text=text)],
    )


def _call(call_id: str, name: str, args: dict):
    # Matches what _extract_function_calls() expects
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(args),
    )


class _MockResponses:
    """
    Mimics client.responses.create
    """
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = []
    

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._sequence.pop(0)
    

class _MockOpenAIClient:
    def __init__(self, sequence):
        self.responses = _MockResponses(sequence)


@pytest.fixture()
def runner_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("AGENT_MAX_SECONDS", "60")
    monkeypatch.setenv("MAX_TOOLS_ITERATIONS", "6")
    monkeypatch.setenv("AGENT_TRACE_LOGS", "false")

    m = importlib.import_module("src.agent.runner")
    return m


# -----------------------
# Tests
# -----------------------
def test_run_task_basic(runner_module, monkeypatch):
    test_client = _MockOpenAIClient(sequence=[
        SimpleNamespace(output=[_msg("Hello, world!")])
    ])

    monkeypatch.setattr(runner_module, "OpenAI", lambda api_key: test_client)

    module_output = runner_module.run_task("hello")
    assert module_output["final_answer"] == "Hello, world!"
    assert module_output["tool_calls"] == []
    assert module_output["metrics"]["openai_calls"] == 1


def test_run_task_basic_with_tool(runner_module, monkeypatch):
    test_client = _MockOpenAIClient(sequence=[
        SimpleNamespace(output=[_call("call_1", "search_kb", {"query": "crm", "top_k": 2})]),
        SimpleNamespace(output=[_msg("final answer")]),
    ])

    monkeypatch.setattr(runner_module, "OpenAI", lambda api_key: test_client)

    monkeypatch.setattr(
        runner_module,
        "_execute_tool",
        lambda name, args: {"results": [{"id": "KB-001"}]},
    )

    module_output = runner_module.run_task("tell me about crm")

    assert module_output["final_answer"] == "final answer"
    assert len(module_output["tool_calls"]) == 1
    assert module_output["metrics"]["openai_calls"] == 2
    assert module_output["tool_calls"][0]["name"] == "search_kb"
    assert module_output["tool_calls"][0]["arguments"] == {"query": "crm", "top_k": 2}
    assert module_output["tool_calls"][0]["result"]["results"][0]["id"] == "KB-001"

     # Prove orchestration appended tool output into the next model call input
    second_input = test_client.responses.calls[1]["input"]
    tool_output = []
    for m in second_input:
        if m.get("type") == "function_call_output" and m.get("call_id") == "call_1":
            tool_output.append(m)
    
    assert len(tool_output) == 1
    assert json.loads(tool_output[0]["output"])["results"][0]["id"] == "KB-001"



