from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path

from src.app.core.config import get_settings

# Regular expression to tokenize text into words (alphanumeric sequences)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_TROUBLESHOOTING_HINTS = {
    "troubleshoot", "troubleshooting", "failed", "failure", "error", "issue", "incident",
    "broken", "not", "working", "timeout", "rate", "limit", "complaint", "degraded"
}

_STOPWORDS = {
    "a", "an", "the", "to", "and", "or", "of", "in", "on", "for",
    "we", "re", "since", "this", "morning", "what", "should", "can", "you", "me"
}

_SYNONYMS = {
    "hubspot": "crm",
    "salesforce": "crm",
    "ops": "operations",
    "ticket": "ticket",
    "tickets": "ticket",
    "writeback": "writeback",
    "write": "write",
    "back": "back",
    "failing": "failed",
    "failure": "failed",
}

_ESCALATION_HINTS = {
    "ticket", "high", "high_priority", "priority", "operations", "incident", "production", "escalate", "escalation"
}

settings = get_settings()


def _resolve_db_path() -> Path:
    path = Path(settings.KB_PATH)
    if not path.is_absolute():
        path = _BASE_DIR / path
    return path


_DB_PATH = _resolve_db_path()


# ---------------------
# Helper functions & models
# ---------------------
def _tokenize(text: str) -> List[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    out: List[str] = []

    for t in toks:
        t = _SYNONYMS.get(t, t)
        if t in _STOPWORDS:
            continue
        out.append(t)
    
    return out



def _is_troubleshooting_query(query_tokens: List[str]) -> bool:
    q = set(query_tokens)
    return len(q.intersection(_TROUBLESHOOTING_HINTS)) > 0


def _is_escalation_query(query_tokens: List[str]) -> bool:
    q = set(query_tokens)
    return len(q.intersection(_ESCALATION_HINTS)) > 0


@dataclass(frozen=True)
class KBEntry:
    id: str
    title: str
    tags: List[str]
    audience: str
    last_updated: Optional[str]
    content: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'KBEntry':
        return KBEntry(
            id=data['id'],
            title=data['title'],
            tags=data.get('tags', []),
            audience=data.get('audience', ''),
            last_updated=(str(data.get("last_updated")) if data.get("last_updated") else None),
            content=str(data["content"]),
        )

# Cache KB in memory
_KB_CACHE: Optional[List[KBEntry]] = None
_KB_MTIME: Optional[float] = None

# ---------------------
# Core functions
# ---------------------
def _build_snippet(content: str, query_tokens: List[str], snippet_length: int = 220) -> str:
    """
    Build a short snippet from content.
    """
    if not content:
        return ""
    
    c = content.strip()
    if not query_tokens:
        # if no query tokens, just return the start of the content
        return c[:snippet_length] + ("..." if len(c) > snippet_length else "")
    lower_c = c.lower()
    first_pos = None
    matched = None

    # Find the first occurrence of any query token- bubble search
    for token in query_tokens:
        pos = lower_c.find(token)
        if pos != -1 and (first_pos is None or pos < first_pos):
            first_pos = pos
            matched = token

    if first_pos is None:
        # no match found, return the start of the content
        return c[:snippet_length] + ("..." if len(c) > snippet_length else "")

    # Build snippet around the first match
    start = max(0, first_pos - snippet_length // 2)
    end = min(len(c), start + snippet_length)
    snippet = c[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(c):
        snippet = snippet + "..."
    
    return snippet


def _load_kb() -> List[KBEntry]:
    global _KB_CACHE, _KB_MTIME
    
    # if data is cached and file not modified, return cache
    mtime = _DB_PATH.stat().st_mtime
    if _KB_CACHE is not None and _KB_MTIME == mtime:
        return _KB_CACHE
    
    data = json.loads(_DB_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("KB data should be a list of entries")
    
    kb_entries = [KBEntry.from_dict(entry) for entry in data]
    _KB_CACHE = kb_entries
    _KB_MTIME = mtime
    return kb_entries


def _score_entry(entry: KBEntry, query_tokens: List[str]) -> int:
    """
    simple keyword matching score:
    - title token hit: +5
    - tag hit: +3
    - content token hit: +1
    Also counts multiple occurrences in content/title via .count.
    """
    title = entry.title.lower()
    content = entry.content.lower()
    tags = entry.tags
    tags = [str(tag.lower()) for tag in tags]

    title_tokens = _tokenize(title)
    content_tokens = _tokenize(content)
    
    score = 0
    for token in query_tokens:
        score += 5 * title_tokens.count(token)
        score += 3 * tags.count(token)
        score += 1 * content_tokens.count(token)
    
    return score


def _soft_preference_bonus(
    entry: KBEntry,
    query_tokens: List[str],
    filters: Optional[Dict[str, Any]],
) -> int:
    """
    Soft preference boosts to improve robustness when the model sends imperfect filters.
    """
    bonus_score = 0
    entry_audience = entry.audience.lower()
    entry_title = entry.title.lower()
    entry_tags = [tag.lower() for tag in entry.tags]

    is_troubleshooting = _is_troubleshooting_query(query_tokens)
    is_escalation = _is_escalation_query(query_tokens)

    if is_troubleshooting and entry_audience == "internal":
        bonus_score += 4
    
    else:
        if entry_audience == "customer":
            bonus_score += 2
    
    if is_escalation:
        if "escalation" in entry_title:
            bonus_score += 8
        if "operations" in entry_tags or "sla" in entry_tags:
            bonus_score += 4
        if entry_audience == "internal":
            bonus_score += 1

    if not filters:
        return bonus_score
    
    requested_audience_val = filters.get("audience")
    if requested_audience_val.strip():
        requested_audience = requested_audience_val.lower().strip()

        if (is_troubleshooting and requested_audience == "internal") or (not is_troubleshooting and requested_audience == "customer"):
            if entry_audience == requested_audience:
                bonus_score += 2
    
    requested_tags = filters.get("tags")
    if requested_tags:
        need = {tag.lower() for tag in requested_tags if str(tag).strip()}
        have = {tag.lower() for tag in entry.tags}
        overlap = len(need.intersection(have))
        bonus_score += 1 * overlap
    
    return bonus_score


# ---------------------
# Search function
# ---------------------
def search_kb(query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search the KB for relevant entries.
    
    Args:
        query: The search query string.
        top_k: Number of top results to return.
        filters: Optional filters to apply (e.g., audience, tags).
    Returns:
        A dictionary with search results.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    
    query_tokens = _tokenize(query)
    kb = _load_kb()
    effective_top_k = top_k if top_k is not None else settings.KB_TOP_K_DEFAULT
    effective_top_k = max(1, min(effective_top_k, 10))  # cap at 10
    scored_entries: List[Tuple[KBEntry, int]] = []

    for entry in kb:
        base = _score_entry(entry, query_tokens)
        if base == 0:
            continue
        
        score = base + _soft_preference_bonus(entry, query_tokens, filters)
        scored_entries.append((entry, score))
    
    # Sort by score, then most recent last_updated if present
    scored_entries.sort(key=lambda x: (x[1], x[0].last_updated if x[0].last_updated else ""), reverse=True)
    top_entries = scored_entries[:effective_top_k]

    max_score = top_entries[0][1] if top_entries else 0.0
    results = []
    for entry, score in top_entries:
        norm = (score / max_score) if max_score else 0.0
        snippet = _build_snippet(entry.content, query_tokens)
        results.append({
            "id": entry.id,
            "title": entry.title,
            "score": float(round(norm, 6)),
            "snippet": snippet,
            "tags": entry.tags,
        })
    
    return {"results": results}