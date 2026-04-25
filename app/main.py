import argparse
import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import agent
from app.schemas import ClinicalBrief


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    state: str
    brief: ClinicalBrief | None = None


app = FastAPI(title="Clinical Intake Agent")

session_store: dict[str, dict] = {}


@app.get("/health")
async def health():
    mock_mode = os.environ.get("MOCK_LLM", "false").lower() == "true"
    return {"status": "ok", "mock_mode": mock_mode}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply, current_node, brief = agent.process_message(
        request.session_id, request.message, session_store
    )

    brief_dict = None
    if brief:
        brief_dict = brief.model_dump()

    return ChatResponse(reply=reply, state=current_node, brief=brief_dict)


def run_cli():
    print("=" * 60)
    print("Clinical Intake Agent - CLI Mode")
    print("=" * 60)
    print("Type your responses. The intake will end when complete.\n")

    session_id = "cli_session"
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        reply, current_node, brief = agent.process_message(session_id, user_input, session_store)

        print(f"\nAgent: {reply}\n")

        if current_node == "done" and brief:
            print("=" * 60)
            print("CLINICAL INTAKE COMPLETE")
            print("=" * 60)
            print(json.dumps(brief.model_dump(), indent=2))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Intake Agent")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)
