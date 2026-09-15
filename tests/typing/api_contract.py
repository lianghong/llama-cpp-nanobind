"""Static public API contracts, checked by mypy without loading a model."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, assert_type

from llama_cpp import Llama


def generation_types(llm: Llama, stream: bool, logprobs: int | None) -> None:
    assert_type(llm.generate("Hello"), str)
    assert_type(llm.generate("Hello", stream=False), str)
    assert_type(llm.generate("Hello", stream=True), Iterator[str])
    assert_type(llm.generate("Hello", logprobs=0), dict[str, Any])
    assert_type(llm.generate("Hello", logprobs=5), dict[str, Any])
    assert_type(
        llm.generate("Hello", stream=stream, logprobs=logprobs),
        str | Iterator[str] | dict[str, Any],
    )
    assert_type(llm.generate("Hello", speculative=True, n_draft_max=2), str)
    assert_type(
        llm.generate("Hello", speculative=True, n_draft_max=2, stream=True),
        Iterator[str],
    )
    messages = [{"role": "user", "content": "Hello"}]
    assert_type(llm.create_chat_completion(messages), dict[str, Any])
    assert_type(
        llm.create_chat_completion(messages, stream=True),
        Iterator[dict[str, Any]],
    )
    assert_type(
        llm.create_chat_completion(messages, stream=stream),
        dict[str, Any] | Iterator[dict[str, Any]],
    )


async def async_generation_types(
    llm: Llama, stream: bool, logprobs: int | None
) -> None:
    assert_type(await llm.generate_async("Hello"), str)
    assert_type(await llm.generate_async("Hello", stream=False), str)
    assert_type(await llm.generate_async("Hello", stream=True), AsyncIterator[str])
    assert_type(await llm.generate_async("Hello", logprobs=0), dict[str, Any])
    assert_type(await llm.generate_async("Hello", logprobs=5), dict[str, Any])
    assert_type(
        await llm.generate_async("Hello", stream=stream, logprobs=logprobs),
        str | AsyncIterator[str] | dict[str, Any],
    )
    messages = [{"role": "user", "content": "Hello"}]
    assert_type(await llm.create_chat_completion_async(messages), dict[str, Any])
    assert_type(
        await llm.create_chat_completion_async(messages, stream=True),
        AsyncIterator[dict[str, Any]],
    )
    assert_type(
        await llm.create_chat_completion_async(messages, stream=stream),
        dict[str, Any] | AsyncIterator[dict[str, Any]],
    )
