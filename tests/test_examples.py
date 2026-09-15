"""Model-free regression checks for example scripts."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from examples import model_helper_utils, streaming_demo


@pytest.mark.parametrize("demo", ["demo_buffered_streaming", "demo_true_streaming"])
@pytest.mark.parametrize("chunks", [[], ["hello", " world"]])
def test_streaming_demo_timing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    demo: str,
    chunks: list[str],
) -> None:
    model = MagicMock()
    model.generate.return_value = iter(chunks)
    model.generate_stream.return_value = iter(chunks)
    factory = MagicMock()
    factory.return_value.__enter__.return_value = model
    monkeypatch.setattr(streaming_demo, "Llama", factory)
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(
        streaming_demo, "time", SimpleNamespace(time=lambda: next(ticks))
    )

    getattr(streaming_demo, demo)()

    output = capsys.readouterr().out
    assert "Total time:" in output
    if chunks:
        assert "Time to first chunk: 1.000s" in output
    else:
        assert "No output chunks were generated." in output


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("fails", [False, True])
async def test_server_stream_returns_instance_to_owning_pool(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, fails: bool
) -> None:
    pytest.importorskip("fastapi")
    from examples import fastapi_server as server

    async def text_chunks() -> AsyncIterator[str]:
        yield "hello"
        if fails:
            raise RuntimeError("generation failed")

    async def chat_chunks() -> AsyncIterator[dict[str, object]]:
        async for text in text_chunks():
            yield {"choices": [{"delta": {"content": text}}]}

    instance = SimpleNamespace(
        generate_async=AsyncMock(return_value=text_chunks()),
        create_chat_completion_async=AsyncMock(return_value=chat_chunks()),
    )
    owning_pool = SimpleNamespace(
        _checkout_instance=AsyncMock(return_value=instance),
        _return_instance=Mock(),
    )
    monkeypatch.setattr(server, "pool", owning_pool)
    if endpoint == "generate":
        response = await server.generate(
            server.GenerateRequest(prompt="hello", stream=True)
        )
    else:
        response = await server.chat(
            server.ChatRequest(
                messages=[server.ChatMessage(role="user", content="hello")],
                stream=True,
            )
        )
    assert isinstance(response, server.StreamingResponse)
    # Shutdown may clear the global while a response still owns its checkout.
    monkeypatch.setattr(server, "pool", None)
    if fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            _ = [chunk async for chunk in response.body_iterator]
    else:
        assert [chunk async for chunk in response.body_iterator] == ["hello"]
    owning_pool._return_instance.assert_called_once_with(instance)


def test_metadata_detection_preserves_normal_tensor_reading(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    import numpy as np

    path = tmp_path / "test.gguf"
    tensor = np.array([1.0, 2.0], dtype=np.float32)
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_name("Qwen3.5 test")
    writer.add_tensor("test.weight", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    assert model_helper_utils.detect_family(path) == "qwen3"
    reader = gguf.GGUFReader(path)
    assert len(reader.tensors) == 1
    np.testing.assert_array_equal(reader.tensors[0].data, tensor)


def test_gptoss_tool_calls_are_strings() -> None:
    call = (
        'to=functions.search<|channel|>commentary<|message|>{"query":"llama"}<|call|>'
    )
    parsed = model_helper_utils.parse_output(call, "gpt-oss")
    assert parsed.tool_calls == [call]


def test_model_stops_helper_accepts_explicit_text_options() -> None:
    model = MagicMock()
    model.generate.return_value = "hello"
    assert (
        model_helper_utils.generate_with_model_stops(
            model, "Hello", "qwen3", stream=False, logprobs=None
        )
        == "hello"
    )
    model.generate.assert_called_once_with(
        "Hello",
        max_tokens=128,
        stop=["<|im_end|>", "<|im_start|>"],
        stream=False,
        logprobs=None,
    )
