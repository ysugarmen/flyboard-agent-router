from __future__ import annotations

import json
import re

from collections import Counter
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
    "tickets": "ticket",
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


@dataclass(frozen=True)
class _IndexedEntry:
    entry: KBEntry
    title_tf: Counter[str]
    content_tf: Counter[str]
    tags_lc: List[str]
    audience_lc: str


# Cache KB in memory
_KB_INDEX: Optional[List[_IndexedEntry]] = None
_KB_MTIME: Optional[float] = None


def _load_kb_index() -> List[_IndexedEntry]:
    global _KB_INDEX, _KB_MTIME

    mtime = _DB_PATH.stat().st_mtime
    if _KB_INDEX is not None and _KB_MTIME == mtime:
        return _KB_INDEX

    raw = json.loads(_DB_PATH.read_text(encoding="utf-8"))
    entries = [KBEntry.from_dict(entry) for entry in raw]

    indexed: List[_IndexedEntry] = []
    for entry in entries:
        title_tf = Counter(_tokenize(entry.title))
        content_tf = Counter(_tokenize(entry.content))
        tags_lc = [str(tag).lower() for tag in entry.tags]
        audience_lc = (entry.audience or "").lower()

        indexed.append(_IndexedEntry(
            entry=entry,
            title_tf=title_tf,
            content_tf=content_tf,
            tags_lc=tags_lc,
            audience_lc=audience_lc,
        ))
    
    _KB_INDEX = indexed
    _KB_MTIME = mtime
    return indexed

        

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
    positions = [lower_c.find(t) for t in set(query_tokens)]
    positions = [p for p in positions if p >= 0]

    if not positions:
        # if no query tokens found, return the start of the content
        return c[:snippet_length] + ("..." if len(c) > snippet_length else "")

    # Build snippet around the first match
    first_pos = min(positions)
    start = max(0, first_pos - snippet_length // 2)
    end = min(len(c), start + snippet_length)
    snippet = c[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(c):
        snippet = snippet + "..."
    
    return snippet


def _score_entry(indexed_entry: _IndexedEntry, query_tokens: Counter[str]) -> int:
    """
    simple keyword matching score:
    - title token hit: +5
    - tag hit: +3
    - content token hit: +1
    Also counts multiple occurrences in content/title via .count.
    """
    score = 0
    tags = indexed_entry.tags_lc

    for t, query_count in query_tokens.items():
        score += query_count * (5 * indexed_entry.title_tf.get(t, 0))
        score += query_count * (3 * tags.count(t))
        score += query_count * (indexed_entry.content_tf.get(t, 0))

    return score


def _soft_preference_bonus(
    indexed_entry: _IndexedEntry,
    query_tokens: List[str],
    filters: Optional[Dict[str, Any]],
) -> int:
    """
    Soft preference boosts to improve robustness when the model sends imperfect filters.
    """
    bonus_score = 0
    query_set = set(query_tokens)

    if query_set & _TROUBLESHOOTING_HINTS:
        if indexed_entry.audience_lc == "internal":
            bonus_score += 2
    
    if query_set & _ESCALATION_HINTS:
        if "operations" in indexed_entry.tags_lc or "escalation" in indexed_entry.entry.title.lower():
            bonus_score += 2
    
    if not filters:
        return bonus_score
    
    audience_req = str(filters.get("audience", "")).strip().lower()
    if audience_req and indexed_entry.audience_lc == audience_req:
        bonus_score += 2
    
    tags_req = filters.get("tags", [])
    if tags_req:
        need = {tag.lower() for tag in tags_req if str(tag).strip()}
        have = {tag.lower() for tag in indexed_entry.tags_lc}
        bonus_score += len(need & have)
    
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
    query_tf = Counter(query_tokens)

    kb_index = _load_kb_index()

    effective_top_k = int(top_k) if top_k is not None else settings.KB_TOP_K_DEFAULT
    effective_top_k = max(1, min(effective_top_k, 10))  # cap at 10
    scored_entries: List[Tuple[_IndexedEntry, int]] = []

    for idx_entry in kb_index:
        base = _score_entry(idx_entry, query_tf)
        if base == 0:
            continue
        
        score = base + _soft_preference_bonus(idx_entry, query_tokens, filters)
        scored_entries.append((idx_entry, score))
    
    # Sort by score, then most recent last_updated if present
    scored_entries.sort(key=lambda x: (x[1], x[0].entry.last_updated or ""), reverse=True)
    top_entries = scored_entries[:effective_top_k]

    max_score = top_entries[0][1] if top_entries else 0.0
    results: List[Dict[str, Any]] = []

    for idx_entry, score in top_entries:
        norm = (score / max_score) if max_score else 0.0
        entry = idx_entry.entry
        snippet = _build_snippet(entry.content, query_tokens)
        results.append({
            "id": entry.id,
            "title": entry.title,
            "score": float(round(norm, 6)),
            "snippet": snippet,
            "tags": entry.tags,
        })
    
    return {"results": results}