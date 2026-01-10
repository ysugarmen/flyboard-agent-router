from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    customer_id: Optional[str] = None
    language: Optional[str] = None

class ToolCallRecord(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]

class Metrics(BaseModel):
    latency_ms: int
    model: str
    openai_calls: int

class RunResponse(BaseModel):
    trace_id: str
    final_answer: str
    tool_calls: List[ToolCallRecord]
    metrics: Metrics