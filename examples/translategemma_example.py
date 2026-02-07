#!/usr/bin/env python3
"""Example usage of TranslateGemma-27b-it with llama-cpp-nanobind.

TranslateGemma is a specialized translation model from Google based on Gemma 3.
It supports translation across 55 languages using ISO 639-1 language codes.

Reference: https://huggingface.co/google/translategemma-27b-it

Supported Languages (ISO 639-1 codes):
    af (Afrikaans), ar (Arabic), bg (Bulgarian), bn (Bengali), ca (Catalan),
    cs (Czech), da (Danish), de (German), el (Greek), en (English),
    es (Spanish), et (Estonian), fa (Persian), fi (Finnish), fil (Filipino),
    fr (French), gl (Galician), gu (Gujarati), he (Hebrew), hi (Hindi),
    hr (Croatian), hu (Hungarian), id (Indonesian), it (Italian), ja (Japanese),
    kn (Kannada), ko (Korean), lt (Lithuanian), lv (Latvian), mk (Macedonian),
    ml (Malayalam), mr (Marathi), ms (Malay), nl (Dutch), no (Norwegian),
    pl (Polish), pt (Portuguese), ro (Romanian), ru (Russian), sk (Slovak),
    sl (Slovenian), sr (Serbian), sv (Swedish), sw (Swahili), ta (Tamil),
    te (Telugu), th (Thai), tr (Turkish), uk (Ukrainian), ur (Urdu),
    vi (Vietnamese), zh (Chinese)

Regionalized variants are also supported (e.g., "en-US", "pt-BR", "zh-CN").
"""

from llama_cpp.unified import UnifiedLLM

# Language code mappings for common languages
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def get_language_name(code: str) -> str:
    """Get full language name from ISO 639-1 code."""
    base_code = code.split("-")[0].split("_")[0].lower()
    return LANGUAGE_NAMES.get(base_code, code)


def translate(
    llm: UnifiedLLM,
    text: str,
    source_lang: str,
    target_lang: str,
    max_tokens: int = 2048,
) -> str:
    """Translate text using TranslateGemma.

    Args:
        llm: UnifiedLLM instance with TranslateGemma model loaded.
        text: Text to translate.
        source_lang: Source language code (e.g., "en", "ja", "zh-CN").
        target_lang: Target language code (e.g., "ja", "en", "de-DE").
        max_tokens: Maximum tokens for generated translation.

    Returns:
        Translated text.

    Example:
        >>> result = translate(llm, "Hello world", "en", "ja")
        >>> print(result)
    """
    # Format prompt for translation task
    # TranslateGemma understands translation instructions in this format
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)

    prompt = (
        f"Translate the following {source_name} text to {target_name}.\n\n"
        f"Source ({source_lang}): {text}\n\n"
        f"Translation ({target_lang}):"
    )

    return llm.generate(prompt, max_tokens=max_tokens).strip()


def translate_batch(
    llm: UnifiedLLM,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    max_tokens: int = 2048,
) -> list[str]:
    """Translate multiple texts sequentially.

    Args:
        llm: UnifiedLLM instance with TranslateGemma model loaded.
        texts: List of texts to translate.
        source_lang: Source language code.
        target_lang: Target language code.
        max_tokens: Maximum tokens per translation.

    Returns:
        List of translated texts.
    """
    results = []
    for text in texts:
        result = translate(llm, text, source_lang, target_lang, max_tokens)
        results.append(result)
        llm.kv_cache_clear()  # Clear cache between translations
    return results


def print_translation(
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """Pretty print a translation result."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    print(f"  {source_name} ({source_lang}): {source_text}")
    print(f"  {target_name} ({target_lang}): {translated_text}")
    print()


def main() -> None:
    model_path = "models/translategemma-27b-it-Q4_K_S.gguf"

    print("Loading TranslateGemma-27b-it model...")
    print("=" * 70)

    with UnifiedLLM(model_path, n_ctx=4096, verbose=False) as llm:
        print(f"Model: {llm}")
        print(f"Context window: {llm.n_ctx()} tokens")
        print("=" * 70)
        print()

        # ------------------------------------------------------------------
        # Example 1: English to Japanese
        # ------------------------------------------------------------------
        print("[Example 1] English -> Japanese")
        print("-" * 70)
        text_en = "The quick brown fox jumps over the lazy dog."
        result = translate(llm, text_en, "en", "ja")
        print_translation(text_en, result, "en", "ja")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 2: English to Chinese (Simplified)
        # ------------------------------------------------------------------
        print("[Example 2] English -> Chinese (Simplified)")
        print("-" * 70)
        text_en2 = "Artificial intelligence is transforming how we live and work."
        result = translate(llm, text_en2, "en", "zh-CN")
        print_translation(text_en2, result, "en", "zh-CN")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 3: Japanese to English
        # ------------------------------------------------------------------
        print("[Example 3] Japanese -> English")
        print("-" * 70)
        text_ja = "桜の花が満開で、とても美しい春の日です。"
        result = translate(llm, text_ja, "ja", "en")
        print_translation(text_ja, result, "ja", "en")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 4: German to French
        # ------------------------------------------------------------------
        print("[Example 4] German -> French")
        print("-" * 70)
        text_de = "Guten Morgen! Wie geht es Ihnen heute?"
        result = translate(llm, text_de, "de", "fr")
        print_translation(text_de, result, "de", "fr")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 5: Korean to English
        # ------------------------------------------------------------------
        print("[Example 5] Korean -> English")
        print("-" * 70)
        text_ko = "오늘 날씨가 정말 좋네요. 산책하러 가실래요?"
        result = translate(llm, text_ko, "ko", "en")
        print_translation(text_ko, result, "ko", "en")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 6: Spanish to Portuguese (Brazilian)
        # ------------------------------------------------------------------
        print("[Example 6] Spanish -> Portuguese (Brazilian)")
        print("-" * 70)
        text_es = "Me gustaría reservar una mesa para dos personas, por favor."
        result = translate(llm, text_es, "es", "pt-BR")
        print_translation(text_es, result, "es", "pt-BR")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 7: Chinese to English
        # ------------------------------------------------------------------
        print("[Example 7] Chinese -> English")
        print("-" * 70)
        text_zh = "机器学习正在改变我们理解世界的方式。"
        result = translate(llm, text_zh, "zh", "en")
        print_translation(text_zh, result, "zh", "en")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 8: Russian to German
        # ------------------------------------------------------------------
        print("[Example 8] Russian -> German")
        print("-" * 70)
        text_ru = "Добро пожаловать в наш магазин! Чем могу помочь?"
        result = translate(llm, text_ru, "ru", "de")
        print_translation(text_ru, result, "ru", "de")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 9: Multi-sentence translation (English to Italian)
        # ------------------------------------------------------------------
        print("[Example 9] Multi-sentence: English -> Italian")
        print("-" * 70)
        text_long = (
            "Welcome to Rome! The Colosseum is one of the most famous landmarks "
            "in the world. Built nearly 2000 years ago, it could hold up to "
            "80,000 spectators. Today, it remains a symbol of ancient Roman "
            "engineering and culture."
        )
        result = translate(llm, text_long, "en", "it")
        print_translation(text_long, result, "en", "it")
        llm.kv_cache_clear()

        # ------------------------------------------------------------------
        # Example 10: Technical text (English to Japanese)
        # ------------------------------------------------------------------
        print("[Example 10] Technical text: English -> Japanese")
        print("-" * 70)
        text_tech = (
            "The API endpoint accepts POST requests with JSON payloads. "
            "Authentication is handled via Bearer tokens in the Authorization header."
        )
        result = translate(llm, text_tech, "en", "ja")
        print_translation(text_tech, result, "en", "ja")

        print("=" * 70)
        print("Translation examples completed!")
        print("=" * 70)


if __name__ == "__main__":
    main()
