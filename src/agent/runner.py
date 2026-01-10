from __future__ import annotations

import json
import time
import uuid

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from app.core.config import get_settings
from app.utils.logger import get_logger
from tools.search_kb import search_kb
from tools.create_tickets import create_ticket
from tools.followup import schedule_followup

logger = get_logger("agent.runner")

settings = get_settings()


class UpstreamModelError(RuntimeError):
    """
    Raised when OpenAI fails; route layer should return 502 including trace_id.
    """
    def __init__(self, trace_id: str, message: str = "Upstream model error"):
        super().__init__(message)
        self.trace_id = trace_id


def _system_prompt(language: Optional[str]) -> str:
    lang_line = ""
    if language:
        lang_line = f"- Respond in this language if possible: {language}\n"

    return (
        "You are a reliable internal agent router for Flyboard.\n"
        "You must decide when to use tools and when to answer from the KB.\n"
        "Rules:\n"
        "- Do NOT browse the web. Use only the local knowledge base via tools.\n"
        "- If the KB doesn't contain the information, say you don't know and offer to create a ticket.\n"
        f"{lang_line}"
        "- When the user asks to open a ticket or schedule follow-up, use the tools.\n"
        "- Be concise and accurate.\n"
        "- Always end with a concrete checklist and a recommended next action.\n"
        "- Do not ask the user questions unless required to proceed.\n"
        
    )

def _tool_definitions() -> List[dict]:
    """
    JSON schema tool definitions for OpenAI Responses API tool calling. :contentReference[oaicite:5]{index=5}
    """
    return [
        {
            "type": "function",
            "name": "search_kb",
            "description": "Search the internal knowledge base for relevant entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "create_ticket",
            "description": "Create a support ticket for the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["customer_id", "title", "description"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "schedule_followup",
            "description": "Schedule a follow-up with the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "when": {"type": "string", "description": "ISO-8601 datetime or natural language like 'tomorrow 10:00'"},
                    "notes": {"type": "string"},
                },
                "required": ["customer_id", "when"],
                "additionalProperties": False,
            },
        },
    ]


def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes server-side tools.
    """
    if name == "search_kb":
        return search_kb(**args)
    if name == "create_ticket":
        return create_ticket(**args)
    if name == "schedule_followup":
        return schedule_followup(**args)
    raise ValueError(f"Unknown tool: {name}")


def _make_trace_id() -> str:
    # Simple unique ID generator for tracing
    return f"trace_{uuid.uuid4().hex}"


def _extract_text_from_response(resp: Any) -> str:
    """
    Extracts text content from OpenAI response objects.
    """
    # Common pattern: resp.output contains items; message item has .content list with text chunks.
    output = getattr(resp, "output", None) or []
    texts: List[str] = []

    for item in output:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []):
                t = getattr(c, "text", "")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
    
    return "\n".join(texts).strip()


def _extract_function_calls(resp: Any) -> List[Tuple[str, str, str]]:
    """
    Extracts function call records from OpenAI response objects.
    Returns list of (call_id, name, arguments_json_string)
    """
    calls: List[Tuple[str, str, str]] = []
    output = getattr(resp, "output", None) or []

    for item in output:
        if getattr(item, "type", None) == "function_call":
            call_id = getattr(item, "call_id", "")
            name = getattr(item, "name", "")
            arguments = getattr(item, "arguments", "")

            if call_id and name:
                calls.append((call_id, name, arguments))
    
    return calls


def run_task(task: str, customer_id: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates the tool loop:
    - call OpenAI Responses API
    - execute any requested tools
    - feed tool outputs back
    - stop when final answer produced
    
    Returns the exact response JSON expected by POST /v1/agent/run.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    
    trace_id = _make_trace_id()
    t0 = time.time()
    deadline = t0 + settings.AGENT_MAX_SECONDS
    openai_calls = 0
    tool_call_records: List[Dict[str, Any]] = []

    user_text = task.strip()
    if customer_id:
        user_text = f"{user_text}\n\n(customer_id: {customer_id})"
    
    system_prompt = _system_prompt(language)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
    ]

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    model_name = settings.OPENAI_MODEL
    max_tool_iterations = settings.MAX_TOOLS_ITERATIONS

    client = OpenAI(api_key=api_key)
    tools = _tool_definitions()

    logger.info("trace_id=%s model=%s start", trace_id, model_name)

    try:
        tool_iterations = 0

        while True:
            openai_calls += 1
            if time.time() > deadline:
                raise TimeoutError(f"agent exceeded max seconds ({settings.AGENT_MAX_SECONDS})")

            resp = client.responses.create(
                model=model_name,
                input=messages,
                tools=tools,
                tool_choice="auto",
            )

            calls = _extract_function_calls(resp)
            if calls:
                tool_iterations += 1
                if tool_iterations > max_tool_iterations:
                    raise ValueError(f"tool iteration cap exceeded (max {max_tool_iterations})")
            
                for call_id, name, args_json in calls:
                    messages.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": args_json,
                        }
                    )
                    try:
                        args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
                        if not isinstance(args, dict):
                            raise ValueError("tool arguments must be an object")
                        
                    except Exception:
                        args = {}

                    start_tool = time.time()
                    result = _execute_tool(name, args)
                    duration_ms = int((time.time() - start_tool) * 1000)

                    tool_call_records.append({"name": name, "arguments": args, "result": result})
                    logger.info("trace_id=%s tool=%s duration_ms=%d", trace_id, name, duration_ms)
                    if settings.AGENT_TRACE_LOGS:
                        logger.debug("trace_id=%s tool=%s args=%s result=%s", trace_id, name, args, result)

                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result, ensure_ascii=False),
                        }
                    )

                continue 

            final_answer = _extract_text_from_response(resp)
            if not final_answer:
                final_answer = "I couldn't generate a final answer. Please try again."
            
            total_latency_ms = int((time.time() - t0) * 1000)

            return {
                "trace_id": trace_id,
                "final_answer": final_answer,
                "tool_calls": tool_call_records,
                "metrics": {
                    "latency_ms": total_latency_ms,
                    "model": model_name,
                    "openai_calls": openai_calls,
                },
            }
            
    except UpstreamModelError:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "openai" in msg or "rate" in msg or "timeout" in msg or "responses" in msg:
            logger.exception("trace_id=%s upstream error", trace_id)
            raise UpstreamModelError(trace_id=trace_id, message="OpenAI request failed") from e

        logger.exception("trace_id=%s runner error", trace_id)
        raise
