# flyboard-agent-router

This project was implemented first, then Codex was used after implementation to verify that all required instructions were followed and to tighten documentation quality.

## Run instructions

### Prerequisites
- Python 3.10+
- An OpenAI API key available via `OPENAI_API_KEY`

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment
```bash
export OPENAI_API_KEY="your-key"
# Optional overrides:
# export KB_PATH=kb.json
```

### Start the APP 
### (from the project root directory "\flyboard-agent-router")
```bash
uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Health check
```bash
curl http://localhost:8000/health
```

### Agent routes
- `POST /agent/run`: runs the agent router with the provided request payload and returns the agent response.
