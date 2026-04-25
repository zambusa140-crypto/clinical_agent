---
title: Clinical Intake Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Clinical Intake Agent

LangGraph-based conversational agent for pre-visit clinical intake.

## Features

- Conducts pre-visit clinical intake with simulated patient
- Generates structured ClinicalBrief (CC, HPI, ROS)
- Runs as FastAPI web app AND CLI
- Hosted on HuggingFace Spaces via Docker

## Usage

### API Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### CLI Mode
```bash
python -m app.main --cli
```

### Environment Variables

- `MOCK_LLM=true` - Use mock LLM for testing
- `MOCK_LLM=false` - Use real LLM (requires llama-cpp-python)

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send message and get response

## Testing

```bash
pytest tests/test_e2e.py -v
```
