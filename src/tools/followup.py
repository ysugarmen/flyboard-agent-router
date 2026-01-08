import json
import threading
from enum import Enum
from typing import Any, Dict
from pathlib import Path
from datetime import datetime


class FollowupChannel(str, Enum):
    email = "email"
    phone = "phone"
    whatsapp = "whatsapp"


_FOLLOWUP_OUTPUT_PATH = Path(__file__).parent.parent.parent / "followups.jsonl"
_FOLLOEUP_COUNTER_PATH = Path(__file__).parent.parent.parent / "followup_counter.txt"

_LOCK = threading.Lock() # handles logic Counter in case of concurrent followup creation


# ---------------------
# Helper functions
# ---------------------
def _load_counter() -> int:
    if not _FOLLOEUP_COUNTER_PATH.exists():
        return 0
    raw = _FOLLOEUP_COUNTER_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        raise ValueError("Invalid followup counter file content")
    

def _save_counter(counter: int) -> None:
    _FOLLOEUP_COUNTER_PATH.write_text(str(counter), encoding="utf-8")


def _get_next_followup_id() -> str:
    with _LOCK:
        counter = _load_counter() + 1
        _save_counter(counter)
        return f"FUP-{counter:06d}"


def _validate_iso8601(date_str: str) -> str:
    if not isinstance(date_str, str) or not date_str.strip():
        raise ValueError("datetime_iso must be a non-empty ISO 8601 string")

    s = date_str.strip()

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    
    try:
        parsed = datetime.fromisoformat(s)
    
    except ValueError:
        raise ValueError("datetime_iso must be a valid ISO 8601 string")
    
    return parsed.isoformat()


# ---------------------
# Create followup function
# ---------------------
def schedule_followup(
    datetime_iso: str,
    contact: str,
    channel: FollowupChannel | str) -> Dict[str, Any]:

    # validate inputs
    dt_iso = _validate_iso8601(datetime_iso)

    if not isinstance(contact, str) or not contact.strip():
        raise ValueError("contact must be a non-empty string")

    try:
        ch = channel if isinstance(channel, FollowupChannel) else FollowupChannel(channel)
    except ValueError:
        raise ValueError(f"Invalid channel. Supported channels: {[c.value for c in FollowupChannel]}")
    
    followup_id = _get_next_followup_id()

    followup = {
        "id": followup_id,
        "datetime_iso": dt_iso,
        "contact": contact.strip(),
        "channel": ch.value,
        "status": True,
    }

    # append to followups file
    line = json.dumps(followup, ensure_ascii=False) + "\n"
    with _LOCK:
        with _FOLLOWUP_OUTPUT_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    
    return {"scheduled": True, "followup_id": followup_id}