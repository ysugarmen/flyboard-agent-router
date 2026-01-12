import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def kb_json(tmp_path):
    entries = [
        {
            "id": "KB-001",
            "title": "CRM integration basics",
            "tags": ["crm", "integration"],
            "audience": "customer",
            "last_updated": "2024-06-10",
            "content": "Use the CRM integration to sync accounts.",
        },
        {
            "id": "KB-002",
            "title": "Email troubleshooting",
            "tags": ["email"],
            "audience": "internal",
            "last_updated": "2024-06-05",
            "content": "If email fails, check the SMTP settings.",
        },
        {
            "id": "KB-003",
            "title": "CRM escalation guide",
            "tags": ["crm", "operations", "escalation"],
            "audience": "internal",
            "last_updated": "2024-06-12",
            "content": "Escalate CRM issues to the operations team.",
        },
    ]
    kb_path = tmp_path / "kb.json"
    kb_path.write_text(json.dumps(entries), encoding="utf-8")
    return kb_path
