# Tools

Command-line tools for web content fetching and translation.

---

# URL to Markdown (url2md.py)

Fetches web page content and converts it to Markdown format. Mimics a real browser to avoid being blocked.

## Features

- **Browser Mimicking**: Rotates User-Agents, sends realistic headers
- **Main Content Extraction**: Automatically extracts article content
- **Smart Filenames**: Auto-generates filenames from page titles
- **Metadata Preservation**: Captures title, description, source URL
- **Retry Logic**: Automatic retries with exponential backoff
- **Batch Processing**: Process multiple URLs from a file
- **Dry-Run Mode**: Preview what would be fetched

## Requirements

- Python 3.14+
- requests
- markdownify
- beautifulsoup4

## Installation

```bash
pip install requests markdownify beautifulsoup4
```

## Usage

### Basic Usage

```bash
# Fetch single URL (auto-generates filename from page title)
python tools/url2md.py --url https://example.com/article
# -> outputs/Article_Title_Here.md

# Save to specific file
python tools/url2md.py --url https://example.com -o article.md

# Fetch multiple URLs from file
python tools/url2md.py --file urls.txt --output-dir docs/
```

### Output Options

```bash
# Default: saves to outputs/ directory
python tools/url2md.py --url https://example.com

# Specify output directory
python tools/url2md.py --url https://example.com --output-dir articles/

# Overwrite existing files (default: skip)
python tools/url2md.py --url https://example.com --overwrite
```

### Fetch Options

```bash
# Custom timeout and retries
python tools/url2md.py --url https://example.com --timeout 60 --retries 5

# Add delay between requests (for multiple URLs)
python tools/url2md.py --file urls.txt --delay 2.0

# Use specific User-Agent
python tools/url2md.py --url https://example.com --user-agent "MyBot/1.0"
```

### Content Options

```bash
# Keep full page (don't extract main content)
python tools/url2md.py --url https://example.com --no-extract-main

# Include links and images (excluded by default for cleaner output)
python tools/url2md.py --url https://example.com --links --images
```

### Dry-Run Mode

```bash
python tools/url2md.py --url https://example.com --dry-run
python tools/url2md.py --file urls.txt --dry-run
```

## Command-Line Options

### Input/Output

| Option | Description |
|--------|-------------|
| `-u, --url URL` | Single URL to fetch |
| `-f, --file PATH` | File containing URLs (one per line) |
| `-o, --output PATH` | Output file or directory |
| `--output-dir PATH` | Output directory (default: `outputs/`) |
| `--output-suffix SUFFIX` | Output file suffix (default: .md) |
| `--overwrite` | Overwrite existing files (default: skip) |

### Fetch Options

| Option | Default | Description |
|--------|---------|-------------|
| `--timeout SEC` | 30 | Request timeout in seconds |
| `--delay SEC` | 1.0 | Delay between requests |
| `--retries N` | 3 | Maximum retry attempts |
| `--user-agent UA` | (rotate) | Custom User-Agent string |

### Content Options

| Option | Description |
|--------|-------------|
| `--no-extract-main` | Don't extract main content (convert full page) |
| `--links` | Include links in output (default: excluded) |
| `--images` | Include images in output (default: excluded) |

### Other Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `--dry-run` | Show what would be fetched |

## URL File Format

```text
# Comments start with #
https://example.com/article1
https://example.com/article2

# Blank lines are ignored
https://example.com/article3
```

## Output Format

Generated Markdown includes:

```markdown
# Page Title

> Source: <https://example.com/article>

> Page description if available

---

[Main content converted to Markdown]
```

## Filename Generation

Filenames are auto-generated from page titles when not specified:

1. **Page title** (preferred): `Article_Title_Here.md`
2. **URL path** (fallback): `article-slug.md`
3. **Domain + hash** (last resort): `example_com_a1b2c3d4.md`

## Workflow with md_translator.py

```bash
# 1. Fetch English article
python tools/url2md.py --url https://example.com/article

# 2. Translate to Chinese
python tools/md_translator.py --file outputs/Article_Title.md --target zh

# Result: outputs/Article_Title.zh.md
```

---

# Markdown Translator (md_translator.py)

A command-line tool for translating Markdown files using local LLM models while preserving formatting, code blocks, and structural elements.

## Features

- **Format Preservation**: Maintains Markdown structure including code blocks, inline code, links, images, tables, and more
- **Multiple Models**: Supports Qwen3-30B-A3B (MoE) and TranslateGemma-27B (Dense) models
- **55+ Languages**: Full ISO 639-1 language support with regional variants
- **Chunked Processing**: Handles large documents by splitting into optimal chunks
- **Dry-Run Mode**: Analyze files without making LLM calls
- **Graceful Shutdown**: Clean interrupt handling with Ctrl+C
- **Configurable Parameters**: Override model defaults for context size, batch size, and sampling

## Requirements

- Python 3.14+
- llama-cpp-nanobind
- GGUF model files

## Installation

The tool is part of the llama-cpp-nanobind package. Ensure you have a compatible model in the `models/` directory:

```bash
# Default models (place in ./models/)
models/Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf
models/translategemma-27b-it-Q4_K_S.gguf
```

## Usage

### Basic Translation

```bash
# Translate a single file (English to Chinese, default)
# Output: outputs/README.zh.md
python tools/md_translator.py --file README.md

# Translate to Japanese
python tools/md_translator.py --file README.md --target ja

# Translate from Japanese to English
python tools/md_translator.py --file document.md --source ja --target en

# Translate entire directory
python tools/md_translator.py --dir docs/ --target de
```

### Output Options

```bash
# Default: saves to outputs/ directory
python tools/md_translator.py --file README.md
# -> outputs/README.zh.md

# Specify output file
python tools/md_translator.py --file README.md -o translated.md

# Specify output directory
python tools/md_translator.py --file README.md --output-dir translations/

# Overwrite existing files (default: skip)
python tools/md_translator.py --file README.md --overwrite
```

### Model Selection

```bash
# Use TranslateGemma (specialized for translation)
python tools/md_translator.py --file README.md --model translategemma

# Use custom model path
python tools/md_translator.py --file README.md --model-path /path/to/model.gguf
```

### Dry-Run Mode

Analyze files without calling the LLM:

```bash
python tools/md_translator.py --file README.md --dry-run
python tools/md_translator.py --dir docs/ --dry-run --target ja
```

Output includes:
- File size and character count
- Estimated token count
- Number of preserved elements (code blocks, links, etc.)
- Chunk breakdown for large files

### View Model Configurations

```bash
python tools/md_translator.py --show-models
```

## Command-Line Options

### Input/Output

| Option | Description |
|--------|-------------|
| `-f, --file PATH` | Single Markdown file to translate |
| `-d, --dir PATH` | Directory containing Markdown files |
| `-o, --output PATH` | Output file or directory |
| `--output-dir PATH` | Output directory (default: `outputs/`) |
| `--output-suffix SUFFIX` | Custom output suffix (default: `.<target_lang>`) |
| `--overwrite` | Overwrite existing files (default: skip) |

### Language Options

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --source LANG` | `en` | Source language code |
| `-t, --target LANG` | `zh` | Target language code |

### Model Selection

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `qwen3` | Model to use (`qwen3` or `translategemma`) |
| `--model-path PATH` | - | Custom model file path (overrides `--model`) |

### Model Parameters

Override model-specific defaults:

| Option | Qwen3 Default | TranslateGemma Default | Description |
|--------|---------------|------------------------|-------------|
| `--n-ctx N` | 10240 | 4096 | Context window size |
| `--n-batch N` | 4096 | 2048 | Batch size for prompt processing |
| `--n-ubatch N` | 512 | 512 | Micro-batch size |
| `--max-tokens N` | 4096 | 2048 | Maximum output tokens |
| `--chunk-tokens N` | 2000 | 1000 | Tokens per chunk for large documents |

### Sampling Parameters

Control generation randomness (lower = more deterministic):

| Option | Qwen3 Default | TranslateGemma Default | Description |
|--------|---------------|------------------------|-------------|
| `--temperature T` | 0.3 | 0.2 | Sampling temperature |
| `--top-p P` | 0.85 | 0.9 | Nucleus sampling threshold |
| `--top-k K` | 30 | 40 | Top-k sampling (0 = disabled) |
| `--min-p P` | 0.05 | 0.0 | Minimum probability threshold |

### Other Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `--dry-run` | Analyze files without calling LLM |
| `--show-models` | Show model configurations and exit |

## Model Comparison

### Qwen3-30B-A3B (Default)

- **Architecture**: Mixture of Experts (MoE) with ~3B active parameters
- **Strengths**: Fast inference, memory efficient, larger context support
- **Best for**: General translation, large documents, batch processing

### TranslateGemma-27B

- **Architecture**: Dense model, all 27B parameters active
- **Strengths**: Specialized for translation, high quality output
- **Best for**: Professional translation, accuracy-critical content
- **Note**: Requires more VRAM, smaller context/chunk sizes

## Supported Languages

The tool supports 55+ languages using ISO 639-1 codes:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| af | Afrikaans | he | Hebrew | pl | Polish |
| ar | Arabic | hi | Hindi | pt | Portuguese |
| bg | Bulgarian | hr | Croatian | ro | Romanian |
| bn | Bengali | hu | Hungarian | ru | Russian |
| ca | Catalan | id | Indonesian | sk | Slovak |
| cs | Czech | it | Italian | sl | Slovenian |
| da | Danish | ja | Japanese | sr | Serbian |
| de | German | kn | Kannada | sv | Swedish |
| el | Greek | ko | Korean | sw | Swahili |
| en | English | lt | Lithuanian | ta | Tamil |
| es | Spanish | lv | Latvian | te | Telugu |
| et | Estonian | mk | Macedonian | th | Thai |
| fa | Persian | ml | Malayalam | tr | Turkish |
| fi | Finnish | mr | Marathi | uk | Ukrainian |
| fil | Filipino | ms | Malay | ur | Urdu |
| fr | French | nl | Dutch | vi | Vietnamese |
| gl | Galician | no | Norwegian | zh | Chinese |
| gu | Gujarati | | | | |

Regional variants are supported (e.g., `en-US`, `pt-BR`, `zh-CN`).

## Preserved Markdown Elements

The following elements are preserved without translation:

- **Code blocks**: Fenced blocks (` ``` ` or ` ~~~ `)
- **Inline code**: Backtick-enclosed text (`` `code` ``)
- **Links**: `[text](url)` and `[text][ref]` formats
- **Images**: `![alt](url)`
- **HTML tags**: `<tag>`, `</tag>`, `<tag/>`
- **Frontmatter**: YAML header (`---...---`)
- **Table delimiters**: `|---|---|` rows
- **Horizontal rules**: `---`, `***`, `___`
- **Task checkboxes**: `[ ]`, `[x]`, `[X]`

Structural markers (headings `#`, lists `-`, blockquotes `>`) are preserved while their content is translated.

## Output Files

Translated files are saved to the `outputs/` directory by default:

```
README.md          -> outputs/README.zh.md      (default)
document.md        -> outputs/document.ja.md    (--target ja)
input.md           -> outputs/input_trans.md    (--output-suffix _trans)
```

## Examples

### Translate Documentation to Multiple Languages

```bash
# Chinese
python tools/md_translator.py --dir docs/ --target zh

# Japanese
python tools/md_translator.py --dir docs/ --target ja

# German
python tools/md_translator.py --dir docs/ --target de
```

### High-Quality Translation

```bash
# Use TranslateGemma with lower temperature for precise translation
python tools/md_translator.py \
    --file important.md \
    --model translategemma \
    --temperature 0.1 \
    --target ja
```

### Large Document with Custom Chunking

```bash
# Increase context and batch size for large files
python tools/md_translator.py \
    --file large-doc.md \
    --n-ctx 16384 \
    --n-batch 4096 \
    --verbose
```

### Preview Before Translation

```bash
# Dry run to see chunk breakdown
python tools/md_translator.py --file README.md --dry-run

# Sample output:
# DRY RUN ANALYSIS REPORT
# ======================================================================
# Source language: English (en)
# Target language: Chinese (zh)
# Model: qwen3
# Chunk size: 2,000 tokens (~8,000 chars)
# ======================================================================
# [1] README.md
#     Output: outputs/README.zh.md
#     Size: 15,234 bytes | 15,000 chars | ~3,750 tokens
#     Placeholders: 45 (code_block=12, inline_code=28, link=5)
#     Chunks: 2
# ======================================================================
```

## Error Handling

- **File not found**: Clear error message with path
- **Model not found**: Suggests downloading to `models/` directory
- **Encoding errors**: Skips file and continues with others
- **Interrupt (Ctrl+C)**: Graceful shutdown, reports progress
- **Invalid n_ubatch > n_batch**: Error with fix suggestion
- **n_batch > n_ctx**: Warning, automatically capped to n_ctx

## Performance Tips

1. **Use Qwen3 for batch processing**: MoE architecture is more memory efficient
2. **Use TranslateGemma for quality**: Specialized model produces better translations
3. **Adjust context size**: Smaller `--n-ctx` reduces VRAM usage
4. **Use dry-run first**: Verify chunk breakdown before processing large directories
5. **Lower temperature**: Values 0.1-0.3 produce more consistent translations
