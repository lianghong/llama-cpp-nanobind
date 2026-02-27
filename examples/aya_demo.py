#!/usr/bin/env python3
"""Demo of Cohere Tiny Aya multilingual model using UnifiedLLM wrapper.

Aya is Cohere's massively multilingual generative model supporting 70+
languages. This demo showcases multilingual generation, translation between
language pairs, and multilingual Q&A using the tiny-aya-global quantized
variant.

Key features:
    - 70+ language support (English, Chinese, French, Arabic, Japanese, etc.)
    - Strong cross-lingual transfer for low-resource languages
    - Low temperature (0.3 default) recommended for faithful output

Reference: https://huggingface.co/CohereForAI/aya-expanse-8b
"""

import argparse
import time

from llama_cpp.unified import UnifiedLLM


DEFAULT_MODEL = "models/tiny-aya-global-q8_0.gguf"
SEPARATOR = "=" * 60


def run_example(
    llm: UnifiedLLM,
    title: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 256,
) -> None:
    """Run a single generation example with timing."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(f"Prompt: {prompt}")
    if system_prompt:
        print(f"System: {system_prompt}")

    llm.kv_cache_clear()
    start = time.perf_counter()
    response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start

    print(f"\nResponse:\n{response}")
    print(f"\n  [{elapsed:.2f}s]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cohere Tiny Aya multilingual demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument(
        "--ctx", type=int, default=4096, help="Context size (default: 4096)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Context size:  {args.ctx}")

    with UnifiedLLM(args.model, n_ctx=args.ctx, verbose=args.verbose) as llm:
        print(f"Model family:  {llm.family.name}")
        print(f"Context window: {llm.n_ctx()} tokens")
        print(f"Temperature:   {llm.model_config.temperature}")

        # -- Example 1: English generation ------------------------------------
        run_example(
            llm,
            "1. English Generation",
            "Explain the concept of machine learning in two sentences.",
        )

        # -- Example 2: Chinese generation ------------------------------------
        run_example(
            llm,
            "2. Chinese Generation",
            "用中文简要介绍量子计算的基本原理。",
        )

        # -- Example 3: English to French translation -------------------------
        run_example(
            llm,
            "3. Translation: English -> French",
            "Translate the following English text to French:\n\n"
            '"The best way to predict the future is to invent it."',
            system_prompt="You are a professional translator. "
            "Provide only the translation, no explanations.",
        )

        # -- Example 4: Arabic generation -------------------------------------
        run_example(
            llm,
            "4. Arabic Generation",
            "اكتب فقرة قصيرة عن أهمية التعليم.",
        )

        # -- Example 5: Japanese to English translation -----------------------
        run_example(
            llm,
            "5. Translation: Japanese -> English",
            "Translate the following Japanese text to English:\n\n"
            '"桜の季節は日本で最も美しい時期の一つです。"',
            system_prompt="You are a professional translator. "
            "Provide only the translation, no explanations.",
        )

        # -- Example 6: Multilingual Q&A -------------------------------------
        run_example(
            llm,
            "6. Multilingual Q&A (French question, French answer)",
            "Quelle est la capitale de l'Allemagne et pourquoi "
            "est-elle historiquement importante?",
            max_tokens=512,
        )

        print(f"\n{SEPARATOR}")
        print("Aya multilingual demo completed.")
        print(SEPARATOR)


if __name__ == "__main__":
    main()
