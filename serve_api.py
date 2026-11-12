"""FastAPI server for serving fine-tuned LLM models and adapters."""

import argparse
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from core.inference_engine import InferenceEngine
from utils.file_manager import list_adapters

app = FastAPI(
    title="LLM Fine-Tuning Studio API",
    description="OpenAI-compatible REST API for serving fine-tuned base models and adapters.",
    version="1.0.0",
)

engine = InferenceEngine()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Base model name or local path")
    adapter: Optional[str] = Field(None, description="Path or name of trained adapter")
    messages: List[ChatMessage] = Field(..., description="Conversation message history")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=4096)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/v1/models")
def list_models():
    adapters = list_adapters()
    return {
        "object": "list",
        "data": [
            {"id": a["name"], "type": "adapter", "size": a["size"], "path": a["path"]}
            for a in adapters
        ],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(req: ChatCompletionRequest):
    # Ensure model is loaded
    if engine.current_model_id != req.model or engine.current_adapter_path != req.adapter:
        success, msg = engine.load_model(req.model, adapter_path=req.adapter)
        if not success:
            raise HTTPException(status_code=400, detail=msg)

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    gen_result = engine.generate_response(
        messages=raw_messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    if not gen_result.get("success"):
        raise HTTPException(status_code=500, detail=gen_result.get("response"))

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": gen_result["response"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": gen_result["tokens_count"],
            "total_tokens": gen_result["tokens_count"],
        },
        "studio_telemetry": {
            "latency_sec": gen_result["latency_sec"],
            "tokens_per_sec": gen_result.get("tokens_per_sec", 0.0),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Fine-Tuning Studio API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    uvicorn.run("serve_api:app", host=args.host, port=args.port, reload=False)

