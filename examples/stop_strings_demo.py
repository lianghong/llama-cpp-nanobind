#!/usr/bin/env python3
"""Demonstrate stop string control for generation."""
from llama_cpp import Llama, LlamaConfig

# Initialize model with context manager for proper cleanup
config = LlamaConfig(model_path="models/Qwen3-8B-Q6_K.gguf", verbose=False)

with Llama("models/Qwen3-8B-Q6_K.gguf", config=config) as llm:
    print("=== Stop at specific word ===")
    text = llm.generate(
        "List colors: red, blue, green,",
        max_tokens=50,
        stop=["yellow"],  # Stop when "yellow" is generated
    )
    print(f"Result: {text}")

    print("\n=== Stop at punctuation ===")
    text = llm.generate(
        "The capital of France is", max_tokens=50, stop=["."]  # Stop at first period
    )
    print(f"Result: {text}")

    print("\n=== Multiple stop strings ===")
    text = llm.generate(
        "Count: 1, 2, 3,",
        max_tokens=50,
        stop=["5", "ten", "\n"],  # Stop at any of these
    )
    print(f"Result: {text}")

    print("\n=== Stop at token ID ===")
    eos_token = llm.token_eos()
    text = llm.generate(
        "Hello world", max_tokens=50, stop=[eos_token]
    )  # Stop at EOS token
    print(f"Result: {text}")

    print("\n=== Multi-token stop sequence ===")
    text = llm.generate(
        "Write a story: Once upon a time",
        max_tokens=100,
        stop=["The End", "END"],  # Stop at multi-token sequences
    )
    print(f"Result: {text[:200]}...")

    print("\n=== Chat with stop strings ===")
    response = llm.create_chat_completion(
        [{"role": "user", "content": "List 3 animals"}],
        max_tokens=50,
        stop=["4.", "4)"],  # Stop before listing 4th item
    )
    print(f"Result: {response['choices'][0]['message']['content']}")
