#!/usr/bin/env python3
"""FastAPI server with thread-safe async generation endpoints."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_cpp import Llama, LlamaConfig

# Global model instance
llm: Llama | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global llm
    config = LlamaConfig(
        model_path="models/Qwen3-8B-Q6_K.gguf",
        n_ctx=4096,
        verbose=True,
    )
    llm = Llama("models/Qwen3-8B-Q6_K.gguf", config=config)
    yield
    # Proper cleanup using close()
    if llm is not None:
        llm.close()
        llm = None


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.8
    stream: bool = False


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text (thread-safe async)."""
    if request.stream:

        async def stream_response():
            async for chunk in await llm.generate_async(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            ):
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/plain")

    text = await llm.generate_async(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {"text": text}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion (thread-safe async)."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:

        async def stream_response():
            async for chunk in await llm.create_chat_completion_async(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            ):
                yield chunk["choices"][0]["delta"].get("content", "")

        return StreamingResponse(stream_response(), media_type="text/plain")

    return await llm.create_chat_completion_async(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
