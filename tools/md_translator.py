#!/usr/bin/env python3
# File              : md_translator.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2026-01-29
# Last Modified Date: 2026-01-29
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
"""Markdown translation CLI tool using llama-cpp-nanobind.

Translates Markdown files while preserving formatting, code blocks,
and structural elements. Supports Qwen3 and TranslateGemma models.

Usage:
    python tools/md_translator.py --file README.md --target ja
    python tools/md_translator.py --dir docs/ --target zh --source en
    python tools/md_translator.py --file input.md --model translategemma
    python tools/md_translator.py --file input.md --model-path /path/to/model.gguf

Requirements:
    - Python 3.14+
    - llama-cpp-nanobind
    - GGUF model files in models/ directory
"""

import argparse
import gc
import re
import signal
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Final, Self, TypedDict

from llama_cpp import SamplingParams
from llama_cpp.unified import UnifiedLLM

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type alias for signal handlers (callable, SIG_DFL, SIG_IGN, or None)
type SignalHandler = Callable[[int, FrameType | None], None] | int | None


class DryRunResult(TypedDict):
    """Type definition for dry-run analysis result."""

    file_path: Path
    output_path: Path
    file_size: int
    char_count: int
    estimated_tokens: int
    extracted_char_count: int
    placeholders: int
    chunks: int
    chunk_details: list[tuple[int, int, int]]
    placeholder_types: dict[str, int]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR: Final[Path] = Path("../models")
DEFAULT_MODEL: Final[str] = "qwen3"
QWEN_MODEL_FILE: Final[str] = "Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf"
TRANSLATEGEMMA_MODEL_FILE: Final[str] = "translategemma-27b-it-Q4_K_S.gguf"


# ---------------------------------------------------------------------------
# Model-Specific Parameter Configurations
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ModelParams:
    """Model-specific inference parameters.

    Attributes:
        model_file: GGUF model filename.
        n_ctx: Context window size (tokens).
        n_batch: Batch size for prompt processing.
        n_ubatch: Micro-batch size for memory efficiency.
        max_tokens: Maximum tokens for generation output.
        chunk_tokens: Token limit per chunk when splitting large documents.
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling (0 = disabled).
        min_p: Minimum probability threshold.
        description: Human-readable model description.
    """

    model_file: str
    n_ctx: int
    n_batch: int
    n_ubatch: int
    max_tokens: int
    chunk_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    description: str


# Qwen3-30B-A3B: MoE model with ~3B active params
# - Memory efficient due to sparse activation
# - Can handle larger context windows
# - Good balance of speed and quality
# - Lower temperature for accurate translation
QWEN3_PARAMS: Final[ModelParams] = ModelParams(
    model_file=QWEN_MODEL_FILE,
    n_ctx=10240,  # Larger context for MoE efficiency
    n_batch=4096,  # Must be > chunk_tokens + prompt overhead
    n_ubatch=512,  # Standard micro-batch
    max_tokens=4096,  # Output limit for translation
    chunk_tokens=2000,  # Chunk size (chunk + prompt < n_batch)
    temperature=0.3,  # Low for deterministic translation
    top_p=0.85,  # Focused sampling
    top_k=30,  # Moderate diversity
    min_p=0.05,  # Filter low-probability tokens
    description="Qwen3-30B-A3B MoE (3B active) - Fast, memory efficient",
)

# TranslateGemma-27B: Dense model specialized for translation
# - All 27B params active = higher VRAM per token
# - Smaller context to reduce memory pressure
# - Optimized for translation quality
# - Uses Google's recommended sampling for translation
TRANSLATEGEMMA_PARAMS: Final[ModelParams] = ModelParams(
    model_file=TRANSLATEGEMMA_MODEL_FILE,
    n_ctx=4096,  # Conservative context for dense 27B model
    n_batch=2048,  # Must be > chunk_tokens + prompt overhead
    n_ubatch=512,  # Micro-batch for memory efficiency
    max_tokens=2048,  # Output limit (must fit in context with input)
    chunk_tokens=1000,  # Small chunks: input + output must fit in n_ctx
    temperature=0.2,  # Very low for precise translation
    top_p=0.9,  # Slightly wider for natural phrasing
    top_k=40,  # Standard for translation
    min_p=0.0,  # Disabled (Gemma default)
    description="TranslateGemma-27B Dense - High quality translation",
)

# Model parameter registry
MODEL_PARAMS: Final[dict[str, ModelParams]] = {
    "qwen3": QWEN3_PARAMS,
    "translategemma": TRANSLATEGEMMA_PARAMS,
}

# Language code mappings
LANGUAGE_NAMES: Final[dict[str, str]] = {
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


class ModelType(StrEnum):
    """Supported model types for translation.

    Each model type corresponds to a specific GGUF model file and
    has associated optimal inference parameters defined in MODEL_PARAMS.

    Attributes:
        QWEN3: Qwen3-30B-A3B MoE model, fast and memory efficient.
        TRANSLATEGEMMA: TranslateGemma-27B dense model, high quality translation.
    """

    QWEN3 = "qwen3"
    TRANSLATEGEMMA = "translategemma"


# ---------------------------------------------------------------------------
# Markdown Block Types
# ---------------------------------------------------------------------------


class BlockType(StrEnum):
    """Types of Markdown blocks for preservation during translation.

    These block types are identified during parsing and replaced with
    placeholders to prevent translation. After translation, the original
    content is restored.

    Attributes:
        TEXT: Regular translatable text content.
        CODE_BLOCK: Fenced code blocks (```...``` or ~~~...~~~).
        INLINE_CODE: Inline code spans (`...`).
        LINK: Markdown links ([text](url) or [text][ref]).
        IMAGE: Markdown images (![alt](url)).
        HTML_TAG: HTML tags (<tag>, </tag>, <tag/>).
        HEADING_MARKER: Heading markers (# through ######).
        LIST_MARKER: List item markers (-, *, +, 1.).
        BLOCKQUOTE: Blockquote markers (>).
        TABLE_DELIMITER: Table separator rows (|---|---|).
        HORIZONTAL_RULE: Horizontal rules (---, ***, ___).
        FRONTMATTER: YAML frontmatter (---...---).
        TASK_CHECKBOX: Task list checkboxes ([ ], [x], [X]).
    """

    TEXT = "text"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    LINK = "link"
    IMAGE = "image"
    HTML_TAG = "html_tag"
    HEADING_MARKER = "heading_marker"
    LIST_MARKER = "list_marker"
    BLOCKQUOTE = "blockquote"
    TABLE_DELIMITER = "table_delimiter"
    HORIZONTAL_RULE = "horizontal_rule"
    FRONTMATTER = "frontmatter"
    TASK_CHECKBOX = "task_checkbox"


@dataclass(slots=True, frozen=True)
class MarkdownBlock:
    """Represents a parsed Markdown block.

    Immutable container for a Markdown element that may or may not
    require translation. Used by MarkdownParser to track non-translatable
    elements during the placeholder substitution process.

    Attributes:
        block_type: The type of Markdown element this block represents.
        content: The original raw content of this block.
        translatable: Whether this block should be translated (default: True).
            Set to False for code, links, images, and other structural elements.
    """

    block_type: BlockType
    content: str
    translatable: bool = True


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------

# Regex patterns for Markdown elements (compiled once)
_CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(```[\w]*\n.*?```|~~~[\w]*\n.*?~~~)", re.DOTALL
)
_INLINE_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(`[^`\n]+`)")
_LINK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\[[^\]]*\])"
)
_IMAGE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(!\[[^\]]*\]\([^)]+\))")
_HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(<[^>]+>|</[^>]+>|<[^>]+/>)", re.DOTALL
)
_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(#{1,6}\s+)", re.MULTILINE)
_LIST_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\s*[-*+]\s+|\s*\d+\.\s+)", re.MULTILINE
)
_BLOCKQUOTE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(>\s*)+", re.MULTILINE)
_TABLE_DELIM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\|?[\s:]*-{3,}[\s:]*(?:\|[\s:]*-{3,}[\s:]*)+\|?$", re.MULTILINE
)
_HR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\s*[-*_]{3,}\s*)$", re.MULTILINE)
_FRONTMATTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^---\n.*?\n---\n", re.DOTALL
)
_TASK_CHECKBOX_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\[[ xX]\])")


class MarkdownParser:
    """Parser for extracting translatable text from Markdown while preserving structure.

    This parser identifies non-translatable Markdown elements (code blocks,
    links, images, etc.) and replaces them with unique placeholders. After
    translation, the original elements can be restored.

    The placeholder format uses Unicode brackets (⟦MDBLOCK_N⟧) which are
    unlikely to appear in normal text and survive translation intact.

    Attributes:
        _placeholder_map: Mapping from placeholder strings to original blocks.
        _counter: Counter for generating unique placeholder IDs.

    Example:
        >>> parser = MarkdownParser()
        >>> text = "Hello `code` world"
        >>> extracted = parser.extract_translatable(text)
        >>> # extracted = "Hello ⟦MDBLOCK_0⟧ world"
        >>> translated = translate(extracted)  # "你好 ⟦MDBLOCK_0⟧ 世界"
        >>> result = parser.restore_placeholders(translated)
        >>> # result = "你好 `code` 世界"
    """

    __slots__ = ("_placeholder_map", "_counter")

    def __init__(self) -> None:
        """Initialize parser with empty placeholder map."""
        self._placeholder_map: dict[str, MarkdownBlock] = {}
        self._counter: int = 0

    def _generate_placeholder(self, block: MarkdownBlock) -> str:
        """Generate a unique placeholder for a non-translatable block.

        Args:
            block: The MarkdownBlock to create a placeholder for.

        Returns:
            A unique placeholder string in the format ⟦MDBLOCK_N⟧.
        """
        placeholder = f"⟦MDBLOCK_{self._counter}⟧"
        self._placeholder_map[placeholder] = block
        self._counter += 1
        return placeholder

    def extract_translatable(self, content: str) -> str:
        """Extract translatable text, replacing non-translatable elements with placeholders.

        Args:
            content: Raw Markdown content.

        Returns:
            Content with non-translatable elements replaced by placeholders.
        """
        self._placeholder_map.clear()
        self._counter = 0

        # Order matters: process more complex patterns first

        # 1. Frontmatter (YAML header)
        if match := _FRONTMATTER_PATTERN.match(content):
            block = MarkdownBlock(
                BlockType.FRONTMATTER, match.group(0), translatable=False
            )
            content = _FRONTMATTER_PATTERN.sub(
                self._generate_placeholder(block), content, count=1
            )

        # 2. Fenced code blocks (```...``` or ~~~...~~~)
        def replace_code_block(m: re.Match[str]) -> str:
            block = MarkdownBlock(BlockType.CODE_BLOCK, m.group(0), translatable=False)
            return self._generate_placeholder(block)

        content = _CODE_BLOCK_PATTERN.sub(replace_code_block, content)

        # 3. Inline code (`...`)
        def replace_inline_code(m: re.Match[str]) -> str:
            block = MarkdownBlock(BlockType.INLINE_CODE, m.group(0), translatable=False)
            return self._generate_placeholder(block)

        content = _INLINE_CODE_PATTERN.sub(replace_inline_code, content)

        # 4. Images (![alt](url))
        def replace_image(m: re.Match[str]) -> str:
            block = MarkdownBlock(BlockType.IMAGE, m.group(0), translatable=False)
            return self._generate_placeholder(block)

        content = _IMAGE_PATTERN.sub(replace_image, content)

        # 5. Links ([text](url) or [text][ref])
        def replace_link(m: re.Match[str]) -> str:
            block = MarkdownBlock(BlockType.LINK, m.group(0), translatable=False)
            return self._generate_placeholder(block)

        content = _LINK_PATTERN.sub(replace_link, content)

        # 6. HTML tags
        def replace_html(m: re.Match[str]) -> str:
            block = MarkdownBlock(BlockType.HTML_TAG, m.group(0), translatable=False)
            return self._generate_placeholder(block)

        content = _HTML_TAG_PATTERN.sub(replace_html, content)

        # 7. Horizontal rules
        def replace_hr(m: re.Match[str]) -> str:
            block = MarkdownBlock(
                BlockType.HORIZONTAL_RULE, m.group(0), translatable=False
            )
            return self._generate_placeholder(block)

        content = _HR_PATTERN.sub(replace_hr, content)

        # 8. Table delimiters
        def replace_table_delim(m: re.Match[str]) -> str:
            block = MarkdownBlock(
                BlockType.TABLE_DELIMITER, m.group(0), translatable=False
            )
            return self._generate_placeholder(block)

        content = _TABLE_DELIM_PATTERN.sub(replace_table_delim, content)

        # 9. Task list checkboxes ([ ], [x], [X])
        def replace_task_checkbox(m: re.Match[str]) -> str:
            block = MarkdownBlock(
                BlockType.TASK_CHECKBOX, m.group(0), translatable=False
            )
            return self._generate_placeholder(block)

        content = _TASK_CHECKBOX_PATTERN.sub(replace_task_checkbox, content)

        return content

    def restore_placeholders(self, translated: str) -> str:
        """Restore original non-translatable elements from placeholders.

        Args:
            translated: Translated text with placeholders.

        Returns:
            Fully restored Markdown content.
        """
        result = translated
        for placeholder, block in self._placeholder_map.items():
            result = result.replace(placeholder, block.content)
        return result

    @property
    def placeholder_count(self) -> int:
        """Return the number of placeholders extracted."""
        return len(self._placeholder_map)

    def get_placeholder_types(self) -> dict[str, int]:
        """Return counts of placeholders by block type.

        Returns:
            Dictionary mapping block type names to counts.
        """
        counts: dict[str, int] = {}
        for block in self._placeholder_map.values():
            type_name = block.block_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TranslationConfig:
    """Configuration for translation operations.

    Encapsulates all settings needed for a translation session including
    language pair, model selection, and output preferences.

    Attributes:
        source_lang: ISO 639-1 source language code (e.g., "en", "ja").
        target_lang: ISO 639-1 target language code (e.g., "zh", "de").
        model_type: The model type to use for translation.
        model_params: Model-specific inference parameters.
        output_suffix: Custom suffix for output files (default: ".<target_lang>").
        verbose: Enable verbose progress output.
        dry_run: If True, show what would be translated without calling LLM.
    """

    source_lang: str = "en"
    target_lang: str = "zh"
    model_type: ModelType = field(default=ModelType.QWEN3)
    model_params: ModelParams = field(default=QWEN3_PARAMS)
    output_suffix: str = ""
    verbose: bool = False
    dry_run: bool = False

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for generation output.

        Returns:
            The max_tokens value from model_params.
        """
        return self.model_params.max_tokens

    @property
    def chunk_tokens(self) -> int:
        """Token limit per chunk when splitting large documents.

        Returns:
            The chunk_tokens value from model_params.
        """
        return self.model_params.chunk_tokens


def get_language_name(code: str) -> str:
    """Get full language name from ISO 639-1 code.

    Handles regional variants by extracting the base language code.
    Falls back to returning the original code if not found.

    Args:
        code: ISO 639-1 language code, optionally with region
            (e.g., "en", "en-US", "zh_CN").

    Returns:
        Full language name (e.g., "English", "Chinese") or the
        original code if not found in LANGUAGE_NAMES.

    Example:
        >>> get_language_name("en")
        'English'
        >>> get_language_name("zh-CN")
        'Chinese'
        >>> get_language_name("unknown")
        'unknown'
    """
    base_code = code.split("-")[0].split("_")[0].lower()
    return LANGUAGE_NAMES.get(base_code, code)


class MarkdownTranslator:
    """Translates Markdown content using LLM while preserving formatting.

    This translator handles the full pipeline of Markdown translation:
    1. Parse and extract non-translatable elements (code, links, etc.)
    2. Split large documents into manageable chunks
    3. Translate each chunk using the configured LLM
    4. Restore original non-translatable elements
    5. Combine chunks into final output

    The translator is designed to be used as a context manager for proper
    resource management, though the LLM cleanup is handled externally.

    Attributes:
        _llm: The UnifiedLLM instance used for translation.
        _config: Translation configuration settings.
        _parser: MarkdownParser instance for placeholder management.
        _closed: Whether the translator has been closed.

    Example:
        >>> config = TranslationConfig(source_lang="en", target_lang="ja")
        >>> with UnifiedLLM(model_path) as llm:
        ...     with MarkdownTranslator(config, llm) as translator:
        ...         result = translator.translate_markdown("# Hello World")
    """

    __slots__ = ("_llm", "_config", "_parser", "_closed")

    def __init__(self, config: TranslationConfig, llm: UnifiedLLM) -> None:
        """Initialize translator with configuration and LLM instance.

        Args:
            config: Translation configuration including language pair and model params.
            llm: UnifiedLLM instance (must already be loaded and initialized).
        """
        self._llm = llm
        self._config = config
        self._parser = MarkdownParser()
        self._closed = False

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and release resources.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.close()

    def close(self) -> None:
        """Release translator resources.

        Marks the translator as closed. Note that the underlying LLM
        cleanup is handled by its own context manager, not here.
        """
        if self._closed:
            return
        self._closed = True

    def _build_translation_prompt(self, text: str) -> str:
        """Build the translation prompt for the LLM.

        Args:
            text: Text to translate (with placeholders preserved).

        Returns:
            Formatted prompt string.
        """
        source_name = get_language_name(self._config.source_lang)
        target_name = get_language_name(self._config.target_lang)

        match self._config.model_type:
            case ModelType.TRANSLATEGEMMA:
                # TranslateGemma: Simple format to avoid prompt translation
                # The model is trained for direct translation without complex instructions
                return (
                    f"Translate from {source_name} to {target_name}:\n\n"
                    f"{text}\n\n"
                    f"{target_name}:"
                )
            case ModelType.QWEN3:
                # Qwen3 format with system prompt style
                return (
                    f"Translate the following {source_name} Markdown text to {target_name}.\n\n"
                    f"Rules:\n"
                    f"1. Preserve ALL placeholder markers like ⟦MDBLOCK_0⟧ exactly\n"
                    f"2. Do not add explanations or notes\n"
                    f"3. Maintain paragraph structure and line breaks\n"
                    f"4. Output only the translation\n\n"
                    f"Text to translate:\n{text}\n\n"
                    f"Translation:"
                )
            case _:
                raise ValueError(f"Unsupported model type: {self._config.model_type}")

    def translate_text(self, text: str) -> str:
        """Translate a single text string.

        Args:
            text: Plain text or Markdown with placeholders.

        Returns:
            Translated text.
        """
        if not text.strip():
            return text

        prompt = self._build_translation_prompt(text)

        if self._config.verbose:
            print(f"  [Translating {len(text)} chars...]", file=sys.stderr)

        result: str = self._llm.generate(prompt, max_tokens=self._config.max_tokens)

        # Clear KV cache to free VRAM for next operation
        self._llm.kv_cache_clear()

        # Clean up output - remove any prompt artifacts
        result = self._clean_translation_output(result)

        return result.strip()

    def _clean_translation_output(self, text: str) -> str:
        """Clean translation output by removing prompt artifacts.

        Args:
            text: Raw translation output.

        Returns:
            Cleaned translation text.
        """
        # Remove common prompt leakage patterns
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        skip_until_content = True

        for line in lines:
            stripped = line.strip().lower()
            # Skip prompt instruction lines at the start
            if skip_until_content:
                if any(
                    pattern in stripped
                    for pattern in [
                        "translate",
                        "翻译",
                        "重要提示",
                        "important:",
                        "source (",
                        "原文",
                        "do not",
                        "不要",
                        "preserve",
                        "保留",
                        "placeholder",
                        "占位符",
                    ]
                ):
                    continue
                # Found actual content
                if line.strip():
                    skip_until_content = False
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def translate_markdown(self, content: str) -> str:
        """Translate Markdown content while preserving structure.

        Args:
            content: Raw Markdown content.

        Returns:
            Translated Markdown with structure preserved.
        """
        # Extract translatable portions
        extracted = self._parser.extract_translatable(content)

        # Split into chunks to handle large documents
        chunks = self._split_into_chunks(extracted)

        translated_chunks: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            if self._config.verbose:
                print(f"  [Chunk {i}/{len(chunks)}]", file=sys.stderr)

            translated = self.translate_text(chunk)
            translated_chunks.append(translated)

        # Combine and restore
        combined = "\n\n".join(translated_chunks)
        return self._parser.restore_placeholders(combined)

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into manageable chunks for translation.

        Uses model-specific chunk_tokens from configuration.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        # Estimate: ~4 chars per token for mixed content
        max_chars = self._config.chunk_tokens * 4

        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n\n")
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for \n\n

            if current_size + para_size > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# ---------------------------------------------------------------------------
# File Processing
# ---------------------------------------------------------------------------


def iter_markdown_files(path: Path) -> Iterator[Path]:
    """Iterate over Markdown files in a directory.

    Args:
        path: Directory path.

    Yields:
        Paths to .md files.
    """
    yield from sorted(path.glob("**/*.md"))


def process_file(
    file_path: Path,
    translator: MarkdownTranslator,
    config: TranslationConfig,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Process a single Markdown file.

    Args:
        file_path: Path to input file.
        translator: Translator instance.
        config: Translation configuration.
        output_dir: Directory for output files (None = same as input).
        overwrite: If False, skip files where output already exists.

    Returns:
        Path to output file, or None if skipped.
    """
    # Generate output filename
    suffix = config.output_suffix or f".{config.target_lang}"
    output_name = f"{file_path.stem}{suffix}.md"

    if output_dir:
        output_path = output_dir / output_name
    else:
        output_path = file_path.with_stem(f"{file_path.stem}{suffix}")

    # Check if output exists
    if output_path.exists() and not overwrite:
        print(f"Skipping: {file_path} (output exists)", file=sys.stderr)
        return None

    content = file_path.read_text(encoding="utf-8")

    print(f"Translating: {file_path}", file=sys.stderr)

    translated = translator.translate_markdown(content)

    output_path.write_text(translated, encoding="utf-8")

    print(f"  -> {output_path}", file=sys.stderr)

    return output_path


def dry_run_file(
    file_path: Path,
    config: TranslationConfig,
    output_dir: Path | None = None,
) -> DryRunResult:
    """Analyze a file for dry-run mode without translation.

    Performs all preprocessing steps (parsing, placeholder extraction,
    chunking) and reports what would happen during actual translation.

    Args:
        file_path: Path to input Markdown file.
        config: Translation configuration.
        output_dir: Directory for output files (None = same as input).

    Returns:
        DryRunResult containing analysis data including file paths,
        character counts, chunk details, and placeholder statistics.
    """
    content = file_path.read_text(encoding="utf-8")

    # Parse and extract placeholders
    parser = MarkdownParser()
    extracted = parser.extract_translatable(content)
    placeholder_count = parser.placeholder_count

    # Calculate chunking
    chunk_tokens = config.chunk_tokens
    max_chars = chunk_tokens * 4

    if len(extracted) <= max_chars:
        chunks = [extracted]
    else:
        chunks = []
        paragraphs = extracted.split("\n\n")
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2
            if current_size + para_size > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

    # Calculate output path
    suffix = config.output_suffix or f".{config.target_lang}"
    output_name = f"{file_path.stem}{suffix}.md"

    if output_dir:
        output_path = output_dir / output_name
    else:
        output_path = file_path.with_stem(f"{file_path.stem}{suffix}")

    # Build chunk details
    chunk_details = [
        (i + 1, len(chunk), len(chunk) // 4) for i, chunk in enumerate(chunks)
    ]

    return {
        "file_path": file_path,
        "output_path": output_path,
        "file_size": file_path.stat().st_size,
        "char_count": len(content),
        "estimated_tokens": len(content) // 4,
        "extracted_char_count": len(extracted),
        "placeholders": placeholder_count,
        "chunks": len(chunks),
        "chunk_details": chunk_details,
        "placeholder_types": parser.get_placeholder_types(),
    }


def print_dry_run_report(
    results: list[DryRunResult],
    config: TranslationConfig,
) -> None:
    """Print formatted dry-run analysis report to stderr.

    Args:
        results: List of analysis results from dry_run_file().
        config: Translation configuration for context.
    """
    out = sys.stderr
    print("\n" + "=" * 70, file=out)
    print("DRY RUN ANALYSIS REPORT", file=out)
    print("=" * 70, file=out)
    print(
        f"Source language: {get_language_name(config.source_lang)} ({config.source_lang})",
        file=out,
    )
    print(
        f"Target language: {get_language_name(config.target_lang)} ({config.target_lang})",
        file=out,
    )
    print(f"Model: {config.model_type.value}", file=out)
    print(
        f"Chunk size: {config.chunk_tokens:,} tokens (~{config.chunk_tokens * 4:,} chars)",
        file=out,
    )
    print(f"Max output: {config.max_tokens:,} tokens", file=out)
    print("=" * 70, file=out)

    total_chars = 0
    total_chunks = 0
    total_placeholders = 0

    for i, result in enumerate(results, 1):
        file_path = result["file_path"]
        output_path = result["output_path"]
        char_count = result["char_count"]
        extracted_chars = result["extracted_char_count"]
        est_tokens = result["estimated_tokens"]
        placeholders = result["placeholders"]
        chunks = result["chunks"]
        chunk_details = result["chunk_details"]
        placeholder_types = result["placeholder_types"]

        total_chars += char_count
        total_chunks += chunks
        total_placeholders += placeholders

        print(f"\n[{i}] {file_path}", file=out)
        print(f"    Output: {output_path}", file=out)
        print(
            f"    Size: {result['file_size']:,} bytes | {char_count:,} chars | ~{est_tokens:,} tokens",
            file=out,
        )
        print(
            f"    After extraction: {extracted_chars:,} chars (removed {char_count - extracted_chars:,})",
            file=out,
        )
        print(f"    Placeholders: {placeholders}", end="", file=out)
        if placeholder_types:
            types_str = ", ".join(
                f"{k}={v}" for k, v in sorted(placeholder_types.items())
            )
            print(f" ({types_str})", file=out)
        else:
            print(file=out)
        print(f"    Chunks: {chunks}", file=out)
        if chunks > 1:
            for idx, chars, tokens in chunk_details:
                print(
                    f"      Chunk {idx}: {chars:,} chars (~{tokens:,} tokens)", file=out
                )

    print("\n" + "-" * 70, file=out)
    print("SUMMARY", file=out)
    print("-" * 70, file=out)
    print(f"Files to process: {len(results)}", file=out)
    print(f"Total characters: {total_chars:,}", file=out)
    print(f"Total estimated tokens: {total_chars // 4:,}", file=out)
    print(f"Total chunks: {total_chunks}", file=out)
    print(f"Total placeholders: {total_placeholders}", file=out)
    print(f"Estimated LLM calls: {total_chunks}", file=out)
    print("=" * 70, file=out)
    print("\nNo files were modified (dry run mode).", file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# Signal Handling & Resource Cleanup
# ---------------------------------------------------------------------------


class GracefulShutdown:
    """Context manager for graceful shutdown handling.

    Installs signal handlers for SIGINT and SIGTERM that set a flag
    instead of immediately terminating. This allows the main loop to
    complete the current operation and perform proper cleanup.

    The original signal handlers are restored when exiting the context.

    Attributes:
        _shutdown_requested: Flag indicating shutdown was requested.
        _original_handlers: Saved original signal handlers for restoration.
        _llm_ref: Optional reference to LLM for emergency cleanup.

    Example:
        >>> with GracefulShutdown() as shutdown:
        ...     shutdown.register_llm(llm)
        ...     while not shutdown.shutdown_requested:
        ...         process_next_item()
    """

    __slots__ = ("_shutdown_requested", "_original_handlers", "_llm_ref")

    def __init__(self) -> None:
        """Initialize shutdown handler with default state."""
        self._shutdown_requested: bool = False
        self._original_handlers: dict[signal.Signals, SignalHandler] = {}
        self._llm_ref: UnifiedLLM | None = None

    def register_llm(self, llm: UnifiedLLM) -> None:
        """Register LLM instance for reference during shutdown.

        Args:
            llm: The UnifiedLLM instance to track.
        """
        self._llm_ref = llm

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested via signal.

        Returns:
            True if SIGINT or SIGTERM was received, False otherwise.
        """
        return self._shutdown_requested

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle interrupt signal by setting flag for main loop.

        This handler does not call sys.exit() to allow the main execution
        flow to handle cleanup properly.

        Args:
            signum: The signal number received (SIGINT=2, SIGTERM=15).
            frame: The current stack frame (unused).
        """
        print("\n[Shutdown requested, cleaning up...]", file=sys.stderr)
        self._shutdown_requested = True

    def __enter__(self) -> Self:
        """Enter context and install signal handlers.

        Installs custom handlers for SIGINT and SIGTERM, saving the
        original handlers for restoration on exit.

        Returns:
            Self for use in with statement.
        """
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context and restore original signal handlers.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Defines all CLI arguments including input/output options, model selection,
    inference parameters, and sampling parameters.

    Returns:
        Configured ArgumentParser instance ready for parse_args().
    """
    parser = argparse.ArgumentParser(
        prog="md_translator",
        description="Translate Markdown files using local LLM models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file README.md --target ja
  %(prog)s --dir docs/ --target zh --source en
  %(prog)s --file input.md --model translategemma --target de

Supported Languages:
  en (English), ja (Japanese), zh (Chinese), de (German),
  fr (French), es (Spanish), ko (Korean), ru (Russian),
  and 50+ more languages (ISO 639-1 codes)
        """,
    )

    # Input source (mutually exclusive, required unless --show-models)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Single Markdown file to translate",
    )
    input_group.add_argument(
        "-d",
        "--dir",
        type=Path,
        help="Directory containing Markdown files",
    )

    # Language options
    parser.add_argument(
        "-s",
        "--source",
        default="en",
        metavar="LANG",
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="zh",
        metavar="LANG",
        help="Target language code (default: zh)",
    )

    # Model selection
    parser.add_argument(
        "-m",
        "--model",
        choices=[m.value for m in ModelType],
        default=ModelType.QWEN3.value,
        help=f"Model to use (default: {ModelType.QWEN3.value})",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_path",
        help="Output file (single file) or directory (default: outputs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        metavar="SUFFIX",
        help="Custom output suffix (default: .<target_lang>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip)",
    )

    # Model path override
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Custom model file path (overrides --model)",
    )

    # Model parameters (override model defaults)
    model_params_group = parser.add_argument_group(
        "model parameters",
        "Override model-specific defaults. If not specified, uses optimal values for selected model.",
    )
    model_params_group.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        metavar="N",
        help="Context window size (qwen3: 10240, translategemma: 6144)",
    )
    model_params_group.add_argument(
        "--n-batch",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for prompt processing (qwen3: 2048, translategemma: 1024)",
    )
    model_params_group.add_argument(
        "--n-ubatch",
        type=int,
        default=None,
        metavar="N",
        help="Micro-batch size (qwen3: 512, translategemma: 256)",
    )
    model_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max output tokens (qwen3: 5000, translategemma: 3072)",
    )
    model_params_group.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Tokens per chunk for large docs (qwen3: 4000, translategemma: 2500)",
    )

    # Sampling parameters
    sampling_group = parser.add_argument_group(
        "sampling parameters",
        "Control generation randomness. Lower values = more deterministic translation.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature (qwen3: 0.3, translategemma: 0.2)",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        metavar="P",
        help="Nucleus sampling threshold (qwen3: 0.85, translategemma: 0.9)",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="Top-k sampling, 0=disabled (qwen3: 30, translategemma: 40)",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        metavar="P",
        help="Min probability threshold (qwen3: 0.05, translategemma: 0.0)",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Dry run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be translated without calling the LLM",
    )

    # Show model configurations
    parser.add_argument(
        "--show-models",
        action="store_true",
        help="Show model configurations and exit",
    )

    return parser


def resolve_model_config(args: argparse.Namespace) -> tuple[Path, ModelParams]:
    """Resolve model path and parameters from arguments.

    CLI arguments override model defaults when explicitly provided.

    Args:
        args: Parsed arguments.

    Returns:
        Tuple of (model_path, model_params).
    """
    model_type = ModelType(args.model)
    base_params = MODEL_PARAMS[model_type.value]

    model_path = args.model_path or MODELS_DIR / base_params.model_file

    # Apply CLI overrides to model parameters
    # Use CLI value if explicitly provided, else use model default
    n_ctx = args.n_ctx if args.n_ctx is not None else base_params.n_ctx
    n_batch = args.n_batch if args.n_batch is not None else base_params.n_batch
    n_ubatch = args.n_ubatch if args.n_ubatch is not None else base_params.n_ubatch
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else base_params.max_tokens
    )
    chunk_tokens = (
        args.chunk_tokens if args.chunk_tokens is not None else base_params.chunk_tokens
    )

    # Sampling parameter overrides
    temperature = (
        args.temperature if args.temperature is not None else base_params.temperature
    )
    top_p = args.top_p if args.top_p is not None else base_params.top_p
    top_k = args.top_k if args.top_k is not None else base_params.top_k
    min_p = args.min_p if args.min_p is not None else base_params.min_p

    # Validate parameter relationships
    if n_ubatch > n_batch:
        raise ValueError(
            f"n_ubatch ({n_ubatch}) must be <= n_batch ({n_batch}). "
            f"Either increase --n-batch or decrease --n-ubatch."
        )

    if n_batch > n_ctx:
        print(
            f"Warning: n_batch ({n_batch}) > n_ctx ({n_ctx}), capping to {n_ctx}",
            file=sys.stderr,
        )
        n_batch = n_ctx

    # Create updated params with overrides
    params = ModelParams(
        model_file=base_params.model_file,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        max_tokens=max_tokens,
        chunk_tokens=chunk_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        description=base_params.description,
    )

    return model_path, params


def show_model_configs() -> None:
    """Display all available model configurations to stdout.

    Prints a formatted table showing each model's parameters including
    memory settings (context size, batch sizes) and sampling parameters
    (temperature, top_p, top_k, min_p).

    This is called when --show-models flag is provided.
    """
    print("Available Model Configurations:")
    print("=" * 70)
    for name, params in MODEL_PARAMS.items():
        print(f"\n[{name}]")
        print(f"  Description:  {params.description}")
        print(f"  Model file:   {params.model_file}")
        print()
        print("  Memory Parameters:")
        print(f"    Context size: {params.n_ctx:,} tokens")
        print(f"    Batch size:   {params.n_batch:,}")
        print(f"    Micro-batch:  {params.n_ubatch}")
        print(f"    Max output:   {params.max_tokens:,} tokens")
        print(f"    Chunk size:   {params.chunk_tokens:,} tokens")
        print()
        print("  Sampling Parameters:")
        print(f"    Temperature:  {params.temperature}")
        print(f"    Top-p:        {params.top_p}")
        print(f"    Top-k:        {params.top_k}")
        print(f"    Min-p:        {params.min_p}")
    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point for the Markdown translation CLI.

    Parses command-line arguments, initializes the translation model,
    and processes the specified files or directory.

    The function handles:
    - Argument parsing and validation
    - Model loading with configured parameters
    - File discovery (single file or directory scan)
    - Translation with progress reporting
    - Graceful shutdown on interrupt signals
    - Resource cleanup on exit or error

    Returns:
        Exit code: 0 if all files processed successfully,
        1 if any errors occurred or files were skipped.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --show-models
    if args.show_models:
        show_model_configs()
        return 0

    # Validate input path (only if not showing models)
    if not args.file and not args.dir:
        parser.error("one of the arguments -f/--file -d/--dir is required")

    if args.file and not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    if args.dir and not args.dir.is_dir():
        print(f"Error: Directory not found: {args.dir}", file=sys.stderr)
        return 1

    # Resolve model and parameters
    model_path, model_params = resolve_model_config(args)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        print(
            f"Please download the model to the '{MODELS_DIR}' directory.",
            file=sys.stderr,
        )
        return 1

    # Build config with model-specific parameters
    config = TranslationConfig(
        source_lang=args.source,
        target_lang=args.target,
        model_type=ModelType(args.model),
        model_params=model_params,
        output_suffix=args.output_suffix,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Collect files to process
    files: list[Path] = []
    if args.file:
        files.append(args.file)
    else:
        files.extend(iter_markdown_files(args.dir))

    if not files:
        print("No Markdown files found to process.", file=sys.stderr)
        return 0

    # Determine output directory
    # Priority: --output-dir > -o (if directory) > default "outputs/"
    output_dir: Path | None = None
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output_path:
        if args.output_path.is_dir() or (
            len(files) > 1 and not args.output_path.suffix
        ):
            output_dir = args.output_path
        elif len(files) == 1:
            # Single file with explicit output path - handled in process_file
            output_dir = (
                args.output_path.parent
                if args.output_path.parent != Path(".")
                else None
            )
    else:
        # Default to outputs/ directory
        output_dir = Path("outputs")

    # Create output directory if needed
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}", file=sys.stderr)

    # Handle dry-run mode (no LLM loading needed)
    if config.dry_run:
        print(f"Analyzing {len(files)} file(s) in dry-run mode...", file=sys.stderr)
        results: list[DryRunResult] = []
        for file_path in files:
            try:
                result = dry_run_file(file_path, config, output_dir)
                results.append(result)
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
                continue
        print_dry_run_report(results, config)
        return 0

    print(f"Model: {model_path.name}", file=sys.stderr)
    print(f"  {model_params.description}", file=sys.stderr)
    print(
        f"  Context: {model_params.n_ctx:,} | Batch: {model_params.n_batch:,} | "
        f"uBatch: {model_params.n_ubatch} | MaxOut: {model_params.max_tokens:,}",
        file=sys.stderr,
    )
    print(
        f"  Temp: {model_params.temperature} | Top-p: {model_params.top_p} | "
        f"Top-k: {model_params.top_k} | Min-p: {model_params.min_p}",
        file=sys.stderr,
    )
    print(
        f"Translation: {get_language_name(config.source_lang)} -> "
        f"{get_language_name(config.target_lang)}",
        file=sys.stderr,
    )
    print(f"Files to process: {len(files)}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    processed = 0

    with GracefulShutdown() as shutdown:
        try:
            # Load model with model-specific parameters
            with UnifiedLLM(
                str(model_path),
                n_ctx=model_params.n_ctx,
                n_batch=model_params.n_batch,
                n_ubatch=model_params.n_ubatch,
                n_gpu_layers=-1,
                verbose=args.verbose,
            ) as llm:
                shutdown.register_llm(llm)

                # Apply translation-optimized sampling parameters
                llm.llm.sampling = SamplingParams(
                    temperature=model_params.temperature,
                    top_p=model_params.top_p,
                    top_k=model_params.top_k,
                    min_p=model_params.min_p,
                )

                with MarkdownTranslator(config, llm) as translator:
                    for file_path in files:
                        if shutdown.shutdown_requested:
                            break

                        try:
                            result = process_file(
                                file_path,
                                translator,
                                config,
                                output_dir=output_dir,
                                overwrite=args.overwrite,
                            )
                            if result is not None:
                                processed += 1
                        except (
                            OSError,
                            UnicodeDecodeError,
                            ValueError,
                            RuntimeError,
                        ) as e:
                            print(
                                f"Error processing {file_path}: {e}",
                                file=sys.stderr,
                            )
                            if args.verbose:
                                traceback.print_exc()
                            continue

        except KeyboardInterrupt:
            print("\n[Interrupted]", file=sys.stderr)
        except (OSError, RuntimeError) as e:
            print(f"Fatal error: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1
        finally:
            # Force cleanup
            gc.collect()

    print("-" * 70, file=sys.stderr)
    print(f"Processed: {processed}/{len(files)} files", file=sys.stderr)

    return 0 if processed == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
