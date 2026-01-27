#!/usr/bin/env python3
"""Example client for testing async generation."""
import asyncio

import httpx


async def test_generate():
    """Test /generate endpoint."""
    async with httpx.AsyncClient() as client:
        # Non-streaming
        response = await client.post(
            "http://localhost:8000/generate",
            json={"prompt": "What is Python?", "max_tokens": 50},
            timeout=30.0,
        )
        print("Generate:", response.json()["text"])

        # Streaming
        print("\nStreaming generate:")
        async with client.stream(
            "POST",
            "http://localhost:8000/generate",
            json={"prompt": "Count to 5:", "max_tokens": 30, "stream": True},
            timeout=30.0,
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
        print()


async def test_chat():
    """Test /chat endpoint."""
    async with httpx.AsyncClient() as client:
        # Non-streaming
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
            },
            timeout=30.0,
        )
        print("\nChat:", response.json()["choices"][0]["message"]["content"])

        # Streaming
        print("\nStreaming chat:")
        async with client.stream(
            "POST",
            "http://localhost:8000/chat",
            json={
                "messages": [{"role": "user", "content": "Name 3 colors"}],
                "max_tokens": 30,
                "stream": True,
            },
            timeout=30.0,
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
        print()


async def main():
    """Run all tests."""
    print("Testing FastAPI server...")
    await test_generate()
    await test_chat()
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
