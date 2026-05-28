#!/usr/bin/env python3
"""FastAPI server with parallel generation endpoints (LlamaPool-backed).

Each request checks out a private Llama instance from a fixed-size pool, so
concurrent requests run in true parallel rather than being serialized by the
per-instance lock that ``Llama.generate_async`` holds. ``pool_size`` is the
parallelism ceiling; each instance loads the full model, so VRAM usage scales
with ``model_size * pool_size``.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp import LlamaConfig
from llama_cpp import LlamaPool
from llama_cpp import SamplingParams
from pydantic import BaseModel


MODEL_PATH = "models/Qwen3.5-4B-Q4_K_M.gguf"
POOL_SIZE = 2

# Global pool instance
pool: LlamaPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pool on startup, drain in-flight requests on shutdown."""
    global pool
    config = LlamaConfig(
        model_path=MODEL_PATH,
        n_ctx=4096,
        verbose=True,
    )
    pool = LlamaPool(MODEL_PATH, pool_size=POOL_SIZE, config=config)
    try:
        yield
    finally:
        if pool is not None:
            await pool.close_graceful(timeout=30.0)
            pool = None


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
    """Generate text. Non-streaming runs on a pooled instance in parallel."""
    assert pool is not None  # set by lifespan
    sampling = SamplingParams(temperature=request.temperature)
    if request.stream:
        # Manual checkout so we can stream from a held instance, then return it.
        instance = await pool._checkout_instance()  # type: ignore[attr-defined]

        async def stream_response():
            try:
                async for chunk in await instance.generate_async(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    sampling=sampling,
                    stream=True,
                ):
                    yield chunk
            finally:
                pool._return_instance(instance)  # type: ignore[union-attr]

        return StreamingResponse(stream_response(), media_type="text/plain")

    text = await pool.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        sampling=sampling,
    )
    return {"text": text}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion. Non-streaming runs on a pooled instance in parallel."""
    assert pool is not None
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        instance = await pool._checkout_instance()  # type: ignore[attr-defined]

        async def stream_response():
            try:
                async for chunk in await instance.create_chat_completion_async(
                    messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True,
                ):
                    yield chunk["choices"][0]["delta"].get("content", "")
            finally:
                pool._return_instance(instance)  # type: ignore[union-attr]

        return StreamingResponse(stream_response(), media_type="text/plain")

    return await pool.create_chat_completion(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pool_loaded": pool is not None,
        "pool_size": POOL_SIZE if pool is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
