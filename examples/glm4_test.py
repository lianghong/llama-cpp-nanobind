#!/usr/bin/env python3
"""Test GLM-4 model with UnifiedLLM."""

from llama_cpp.unified import UnifiedLLM


def main():
    model_path = "models/THUDM_GLM-4-9B-0414-Q6_K_L.gguf"

    print(f"Loading {model_path}...")

    # Use context manager for proper cleanup
    with UnifiedLLM(model_path, n_ctx=4096, verbose=True) as llm:
        print(f"Model family: {llm.family.name}")
        print(f"Supports thinking: {llm.supports_thinking}")
        print(f"Max context: {llm.model_config.max_ctx}")
        print()

        # Test basic generation
        print("=" * 60)
        print("Test 1: Basic generation")
        print("=" * 60)
        response = llm.generate("What is the capital of China?", max_tokens=30)
        print("Q: What is the capital of China?")
        print(f"A: {response}")
        print()

        # Test with system prompt
        print("=" * 60)
        print("Test 2: With system prompt")
        print("=" * 60)
        response = llm.generate(
            "Explain machine learning briefly.",
            system_prompt="You are a helpful AI assistant.",
            max_tokens=80,
        )
        print("Q: Explain machine learning briefly.")
        print(f"A: {response}")
        print()

        # Test Chinese language support
        print("=" * 60)
        print("Test 3: Chinese language")
        print("=" * 60)
        response = llm.generate("用一句话解释人工智能。", max_tokens=50)
        print("Q: 用一句话解释人工智能。")
        print(f"A: {response}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
