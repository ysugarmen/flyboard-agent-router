from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
from pathlib import Path

# Regular expression to tokenize text into words (alphanumeric sequences)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DB_PATH = Path(__file__).parent.parent.parent / "kb.json"


# ---------------------
# Helper functions & models
# ---------------------
def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


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


def _apply_filters(entry: KBEntry, filters: Optional[Dict[str, Any]] = None) -> bool:
    """
    Apply filters to a KB entry.
    Supported filters:
    - audience: str
    - tags: List[str]
    """
    if not filters:
        return True
    
    # audience filter
    audience = filters.get("audience")
    if audience is not None:
        if entry.audience.lower() != audience.lower():
            return False
    
    # tags filter
    tags = filters.get("tags")
    if tags is not None:
        need = {tag.lower() for tag in tags}
        have = {tag.lower() for tag in entry.tags}
        if not need.issubset(have):
            return False
    
    return True


# ---------------------
# Search function
# ---------------------
def search_kb(query: str, top_k: Optional[int] = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    top_k = max(1, min(top_k, 10))  # cap at 10
    scored_entries: List[Tuple[KBEntry, int]] = []

    for entry in kb:
        if not _apply_filters(entry, filters):
            continue
        score = _score_entry(entry, query_tokens)
        if score > 0:
            scored_entries.append((entry, score))
    
    # Sort by score, then most recent last_updated if present
    scored_entries.sort(key=lambda x: (x[1], x[0].last_updated if x[0].last_updated else ""), reverse=True)
    top_entries = scored_entries[:top_k]

    max_score = top_entries[0][1] if scored_entries else 0.0
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