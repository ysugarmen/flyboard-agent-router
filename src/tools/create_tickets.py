import json
import threading
from enum import Enum
from typing import Any, Dict
from pathlib import Path


class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


_TICKET_OUTPUT_PATH = Path(__file__).parent.parent.parent / "tickets.jsonl"
_TICKET_COUNTER_PATH = Path(__file__).parent.parent.parent / "ticket_counter.txt"

_LOCK = threading.Lock() # handles logic Counter in case of concurrent ticket creation


# ---------------------
# Helper functions & models
# ---------------------
def _load_counter() -> int:
    if not _TICKET_COUNTER_PATH.exists():
        return 0
    raw = _TICKET_COUNTER_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        raise ValueError("Invalid ticket counter file content")


def _save_counter(counter: int) -> None:
    _TICKET_COUNTER_PATH.write_text(str(counter), encoding="utf-8")


def _get_next_ticket_id() -> str:
    with _LOCK:
        counter = _load_counter() + 1
        _save_counter(counter)
        return f"TICK-{counter:06d}"


# ---------------------
# Create ticket function
# ---------------------
def create_ticket(
    title: str,
    body: str,
    priority: TicketPriority | str) -> Dict[str, Any]:

    # validate inputs
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("body must be a non-empty string")
    pr = priority if isinstance(priority, TicketPriority) else TicketPriority(priority)
    

    ticket_id = _get_next_ticket_id()

    ticket = {
        "id": ticket_id,
        "title": title.strip(),
        "body": body.strip(),
        "priority": pr.value,
        "status": "created"
    }
    # append to tickets file
    line = json.dumps(ticket, ensure_ascii=False) + "\n"
    with _LOCK:
        with _TICKET_OUTPUT_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    
    return {"ticket_id": ticket_id, "status": "created"}