#!/usr/bin/env python3
"""URL to Markdown converter CLI tool.

Fetches web page content and converts it to Markdown format.
Mimics a standard browser to avoid being blocked by websites.

Usage:
    python tools/url2md.py --url https://example.com
    python tools/url2md.py --url https://example.com --output article.md
    python tools/url2md.py --file urls.txt --output-dir docs/
    python tools/url2md.py --url https://example.com --dry-run

Requirements:
    - Python 3.14+
    - requests
    - markdownify
    - beautifulsoup4
"""

import argparse
import gc
import hashlib
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Final, Self
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from bs4 import BeautifulSoup
    from bs4.element import Tag

# Type alias for signal handlers
type SignalHandler = Callable[[int, FrameType | None], None] | int | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: Final[Path] = Path(".")
DEFAULT_TIMEOUT: Final[int] = 30
DEFAULT_DELAY: Final[float] = 1.0
MAX_RETRIES: Final[int] = 3
RETRY_DELAY: Final[float] = 2.0

# Common browser User-Agent strings (rotated to avoid detection)
USER_AGENTS: Final[list[str]] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
]

# Browser-like headers
# Note: brotli (br) excluded - requires brotli package for decompression
DEFAULT_HEADERS: Final[dict[str, str]] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FetchConfig:
    """Configuration for URL fetching operations.

    Attributes:
        output_path: Output file path (for single URL) or directory (for multiple).
        output_suffix: Custom suffix for output files.
        timeout: Request timeout in seconds.
        delay: Delay between requests in seconds.
        retries: Maximum retry attempts for failed requests.
        user_agent: Custom User-Agent string (None = rotate randomly).
        verbose: Enable verbose output.
        dry_run: Show what would be fetched without fetching.
        overwrite: Overwrite existing output files.
        extract_main: Try to extract main content only.
        include_links: Include links in Markdown output.
        include_images: Include images in Markdown output.
    """

    output_path: Path | None = None
    output_suffix: str = ".md"
    timeout: int = DEFAULT_TIMEOUT
    delay: float = DEFAULT_DELAY
    retries: int = MAX_RETRIES
    user_agent: str | None = None
    verbose: bool = False
    dry_run: bool = False
    overwrite: bool = False
    extract_main: bool = True
    include_links: bool = False
    include_images: bool = False


# ---------------------------------------------------------------------------
# URL Fetcher
# ---------------------------------------------------------------------------


class BrowserSession:
    """HTTP session that mimics a real browser.

    Handles cookie persistence, User-Agent rotation, and retry logic
    to avoid being blocked by websites.

    Attributes:
        _session: Underlying requests Session object.
        _config: Fetch configuration.
        _user_agent: Current User-Agent string.
    """

    __slots__ = ("_session", "_config", "_user_agent")

    def __init__(self, config: FetchConfig) -> None:
        """Initialize browser session with configuration.

        Args:
            config: Fetch configuration including timeout and retry settings.
        """
        import requests  # type: ignore[import-untyped]

        self._session = requests.Session()
        self._config = config
        self._user_agent = config.user_agent or random.choice(USER_AGENTS)

        # Set up browser-like headers
        self._session.headers.update(DEFAULT_HEADERS)
        self._session.headers["User-Agent"] = self._user_agent

    def fetch(self, url: str) -> tuple[str, str]:
        """Fetch URL content with retry logic.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (html_content, final_url after redirects).

        Raises:
            requests.RequestException: If all retries fail.
        """
        import requests

        last_error: Exception | None = None

        for attempt in range(self._config.retries):
            try:
                if self._config.verbose and attempt > 0:
                    print(
                        f"  [Retry {attempt + 1}/{self._config.retries}]",
                        file=sys.stderr,
                    )

                response = self._session.get(
                    url,
                    timeout=self._config.timeout,
                    allow_redirects=True,
                )
                response.raise_for_status()

                # Detect encoding from headers or content
                if response.encoding is None or response.encoding == "ISO-8859-1":
                    # requests defaults to ISO-8859-1 for text/* without charset
                    # Try to detect from content or use utf-8
                    response.encoding = response.apparent_encoding or "utf-8"

                content = response.text

                # Validate content looks like HTML (not binary/compressed)
                if not self._is_valid_html(content):
                    raise ValueError(
                        "Response content appears to be binary or compressed. "
                        "The server may require different headers or authentication."
                    )

                return content, response.url

            except requests.RequestException as e:
                last_error = e
                if attempt < self._config.retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        raise last_error or RuntimeError(f"Failed to fetch {url}")

    def _is_valid_html(self, content: str) -> bool:
        """Check if content appears to be valid HTML.

        Detects binary/compressed data that wasn't properly decoded.

        Args:
            content: Response content to validate.

        Returns:
            True if content looks like HTML, False otherwise.
        """
        if not content or len(content) < 10:
            return False

        # Check for common HTML markers
        content_lower = content[:1000].lower()
        html_markers = ["<!doctype", "<html", "<head", "<body", "<meta", "<title"]

        if any(marker in content_lower for marker in html_markers):
            return True

        # Check for high ratio of non-printable characters (binary data)
        # More than 10% non-printable = likely binary/compressed
        sample = content[:500]
        non_printable = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
        return non_printable / len(sample) <= 0.1

    def close(self) -> None:
        """Close the session and release resources."""
        self._session.close()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close session."""
        self.close()


# ---------------------------------------------------------------------------
# HTML to Markdown Converter
# ---------------------------------------------------------------------------


class HTMLToMarkdown:
    """Converts HTML content to Markdown format.

    Uses BeautifulSoup for parsing and markdownify for conversion.
    Optionally extracts main content and cleans up the output.

    Attributes:
        _config: Fetch configuration for conversion options.
    """

    __slots__ = ("_config",)

    # Elements that typically contain main content
    MAIN_CONTENT_SELECTORS: Final[list[str]] = [
        "article",
        "main",
        '[role="main"]',
        ".post-content",
        ".article-content",
        ".entry-content",
        ".content",
        "#content",
        ".post",
        ".article",
    ]

    # Elements to remove before conversion
    REMOVE_SELECTORS: Final[list[str]] = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        ".sidebar",
        ".advertisement",
        ".ads",
        ".social-share",
        ".comments",
        ".related-posts",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
    ]

    def __init__(self, config: FetchConfig) -> None:
        """Initialize converter with configuration.

        Args:
            config: Fetch configuration for conversion options.
        """
        self._config = config

    def convert(self, html: str, url: str) -> str:
        """Convert HTML to Markdown.

        Args:
            html: Raw HTML content.
            url: Source URL (used for resolving relative links).

        Returns:
            Markdown formatted content with metadata header.
        """
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        soup = BeautifulSoup(html, "html.parser")

        # Extract metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)

        # Remove unwanted elements
        for selector in self.REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # Extract main content if configured
        if self._config.extract_main:
            main_content = self._extract_main_content(soup)
            if main_content:
                soup = BeautifulSoup(str(main_content), "html.parser")

        # Convert to Markdown
        markdown = markdownify(
            str(soup),
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
        )

        # Clean up the markdown
        markdown = self._clean_markdown(markdown)

        # Build output with metadata header
        output_parts = [
            f"# {title}" if title else "# Untitled",
            "",
            f"> Source: <{url}>",
        ]

        if description:
            output_parts.extend(["", f"> {description}"])

        output_parts.extend(["", "---", "", markdown])

        return "\n".join(output_parts)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML.

        Args:
            soup: Parsed HTML document.

        Returns:
            Page title or empty string if not found.
        """
        # Try og:title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()

        # Try <title> tag
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return str(title_tag.string).strip()

        # Try first <h1>
        h1_tag = soup.find("h1")
        if h1_tag:
            return str(h1_tag.get_text(strip=True))

        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description from HTML.

        Args:
            soup: Parsed HTML document.

        Returns:
            Page description or empty string if not found.
        """
        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return str(og_desc["content"]).strip()

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return str(meta_desc["content"]).strip()

        return ""

    def _extract_main_content(self, soup: BeautifulSoup) -> Tag | None:
        """Try to extract main content element.

        Args:
            soup: Parsed HTML document.

        Returns:
            Main content element or None if not found.
        """
        for selector in self.MAIN_CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                return element
        return None

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted Markdown.

        Args:
            markdown: Raw converted Markdown.

        Returns:
            Cleaned Markdown with normalized whitespace.
        """
        # Remove excessive blank lines (more than 2 consecutive)
        markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in markdown.split("\n")]
        markdown = "\n".join(lines)

        # Remove leading/trailing blank lines
        markdown = markdown.strip()

        # Ensure single newline at end
        markdown += "\n"

        return markdown


# ---------------------------------------------------------------------------
# File Processing
# ---------------------------------------------------------------------------


def generate_filename_from_title(title: str, suffix: str = ".md") -> str | None:
    """Generate a filename from page title.

    Args:
        title: Page title string.
        suffix: File suffix (default: .md).

    Returns:
        Generated filename or None if title is unusable.
    """
    if not title:
        return None

    # Clean up the title for use as filename
    name = title.strip()

    # Remove common suffixes like " | Site Name" or " - Site Name"
    name = re.sub(r"\s*[|\-–—]\s*[^|\-–—]+$", "", name)

    # Replace problematic characters with underscores
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    # Limit length (filesystem limit is usually 255, leave room for suffix)
    if len(name) > 100:
        name = name[:100].rstrip("_")

    if name and len(name) >= 3:
        return f"{name}{suffix}"

    return None


def generate_filename(url: str, suffix: str = ".md") -> str:
    """Generate a filename from URL.

    Uses the URL path and domain to create a readable filename.
    Falls back to a hash if the URL produces no usable name.

    Args:
        url: Source URL.
        suffix: File suffix (default: .md).

    Returns:
        Generated filename.
    """
    parsed = urlparse(url)

    # Try to get a name from the path
    path = parsed.path.rstrip("/")
    if path:
        # Get last path component
        name = path.split("/")[-1]
        # Remove extension if present
        name = re.sub(r"\.[^.]+$", "", name)
        # Clean up the name
        name = re.sub(r"[^\w\-]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        if name:
            return f"{name}{suffix}"

    # Fall back to domain + hash
    domain = parsed.netloc.replace("www.", "")
    domain = re.sub(r"[^\w\-]", "_", domain)
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{domain}_{url_hash}{suffix}"


def extract_title_from_html(html: str) -> str:
    """Extract page title from HTML without full conversion.

    Args:
        html: Raw HTML content.

    Returns:
        Page title or empty string if not found.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Try og:title first (usually cleanest)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return str(og_title["content"]).strip()

    # Try <title> tag
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return str(title_tag.string).strip()

    # Try first <h1>
    h1_tag = soup.find("h1")
    if h1_tag:
        return h1_tag.get_text(strip=True)

    return ""


def iter_urls_from_file(file_path: Path) -> Iterator[str]:
    """Read URLs from a file, one per line.

    Skips empty lines and lines starting with #.

    Args:
        file_path: Path to file containing URLs.

    Yields:
        URLs from the file.
    """
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def process_url(
    url: str,
    session: BrowserSession,
    converter: HTMLToMarkdown,
    config: FetchConfig,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Fetch URL and convert to Markdown file.

    Args:
        url: URL to fetch.
        session: Browser session for fetching.
        converter: HTML to Markdown converter.
        config: Fetch configuration.
        output_path: Explicit output file path (takes precedence).
        output_dir: Directory for auto-generated filenames.

    Returns:
        Path to the created Markdown file, or None if skipped.
    """
    print(f"Fetching: {url}", file=sys.stderr)

    # Fetch content
    html, final_url = session.fetch(url)

    if config.verbose:
        if final_url != url:
            print(f"  [Redirected to: {final_url}]", file=sys.stderr)
        print(f"  [Received {len(html):,} bytes]", file=sys.stderr)

    # Determine output path (before conversion for better error messages)
    if output_path is None:
        # Try to use page title for filename (more readable)
        title = extract_title_from_html(html)
        filename = generate_filename_from_title(title, config.output_suffix)
        if filename is None:
            # Fall back to URL-based filename
            filename = generate_filename(final_url, config.output_suffix)

        output_path = output_dir / filename if output_dir else Path(filename)

    # Check if output exists
    if output_path.exists() and not config.overwrite:
        print(f"  Skipping: {output_path} (exists)", file=sys.stderr)
        return None

    # Convert to Markdown
    markdown = converter.convert(html, final_url)

    if config.verbose:
        print(f"  [Converted to {len(markdown):,} chars]", file=sys.stderr)

    # Write output
    output_path.write_text(markdown, encoding="utf-8")

    print(f"  -> {output_path}", file=sys.stderr)

    return output_path


def dry_run_url(url: str, config: FetchConfig) -> dict[str, str | Path]:
    """Analyze URL for dry-run mode without fetching.

    Args:
        url: URL to analyze.
        config: Fetch configuration.

    Returns:
        Dictionary with analysis results.
    """
    parsed = urlparse(url)
    filename = generate_filename(url, config.output_suffix)

    output_path = config.output_path
    if output_path and output_path.is_dir():
        output_path = output_path / filename
    elif output_path is None:
        output_path = Path(filename)

    return {
        "url": url,
        "domain": parsed.netloc,
        "path": parsed.path or "/",
        "output_path": output_path,
        "filename": filename,
    }


def print_dry_run_report(
    results: list[dict[str, str | Path]],
    config: FetchConfig,
) -> None:
    """Print formatted dry-run analysis report.

    Args:
        results: List of analysis results from dry_run_url().
        config: Fetch configuration for context.
    """
    out = sys.stderr
    print("\n" + "=" * 70, file=out)
    print("DRY RUN ANALYSIS REPORT", file=out)
    print("=" * 70, file=out)
    print(f"Timeout: {config.timeout}s", file=out)
    print(f"Delay between requests: {config.delay}s", file=out)
    print(f"Max retries: {config.retries}", file=out)
    print(f"Extract main content: {config.extract_main}", file=out)
    print("=" * 70, file=out)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['url']}", file=out)
        print(f"    Domain: {result['domain']}", file=out)
        print(f"    Path: {result['path']}", file=out)
        print(f"    Output: {result['output_path']}", file=out)

    print("\n" + "-" * 70, file=out)
    print("SUMMARY", file=out)
    print("-" * 70, file=out)
    print(f"URLs to fetch: {len(results)}", file=out)
    if len(results) > 1:
        total_time = config.delay * (len(results) - 1) + config.timeout * len(results)
        print(f"Estimated max time: {total_time:.1f}s", file=out)
    print("=" * 70, file=out)
    print("\nNo files were created (dry run mode).", file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# Signal Handling
# ---------------------------------------------------------------------------


class GracefulShutdown:
    """Context manager for graceful shutdown handling.

    Identical to md_translator.py implementation for consistency.
    """

    __slots__ = ("_shutdown_requested", "_original_handlers")

    def __init__(self) -> None:
        """Initialize shutdown handler."""
        self._shutdown_requested: bool = False
        self._original_handlers: dict[signal.Signals, SignalHandler] = {}

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle interrupt signal."""
        print("\n[Shutdown requested, cleaning up...]", file=sys.stderr)
        self._shutdown_requested = True

    def __enter__(self) -> Self:
        """Enter context and install signal handlers."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context and restore signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="url2md",
        description="Fetch web pages and convert to Markdown format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://example.com
  %(prog)s --url https://example.com --output article.md
  %(prog)s --file urls.txt --output-dir docs/
  %(prog)s --url https://example.com --dry-run

The tool mimics a real browser to avoid being blocked by websites.
        """,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-u",
        "--url",
        type=str,
        help="Single URL to fetch",
    )
    input_group.add_argument(
        "-f",
        "--file",
        type=Path,
        help="File containing URLs (one per line)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_path",
        help="Output file (single URL) or directory (multiple URLs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for multiple URLs",
    )
    parser.add_argument(
        "--output-suffix",
        default=".md",
        metavar="SUFFIX",
        help="Output file suffix (default: .md)",
    )

    # Fetch options
    fetch_group = parser.add_argument_group(
        "fetch options",
        "Control HTTP request behavior.",
    )
    fetch_group.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SEC",
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    fetch_group.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        metavar="SEC",
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    fetch_group.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        metavar="N",
        help=f"Maximum retry attempts (default: {MAX_RETRIES})",
    )
    fetch_group.add_argument(
        "--user-agent",
        type=str,
        metavar="UA",
        help="Custom User-Agent string (default: rotate browser UAs)",
    )

    # Content options
    content_group = parser.add_argument_group(
        "content options",
        "Control content extraction and conversion.",
    )
    content_group.add_argument(
        "--no-extract-main",
        action="store_true",
        help="Don't try to extract main content (convert full page)",
    )
    content_group.add_argument(
        "--links",
        action="store_true",
        help="Include links in output (default: excluded)",
    )
    content_group.add_argument(
        "--images",
        action="store_true",
        help="Include images in output (default: excluded)",
    )

    # Other options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without fetching",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip)",
    )

    return parser


def main() -> int:
    """Main entry point.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate file input
    if args.file and not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    # Build configuration
    config = FetchConfig(
        output_path=args.output_path,
        output_suffix=args.output_suffix,
        timeout=args.timeout,
        delay=args.delay,
        retries=args.retries,
        user_agent=args.user_agent,
        verbose=args.verbose,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        extract_main=not args.no_extract_main,
        include_links=args.links,
        include_images=args.images,
    )

    # Collect URLs
    urls: list[str] = []
    if args.url:
        urls.append(args.url)
    else:
        urls.extend(iter_urls_from_file(args.file))

    if not urls:
        print("No URLs to process.", file=sys.stderr)
        return 0

    # Determine output directory
    # Priority: --output-dir > -o (if directory) > default "outputs/"
    output_dir: Path | None = None
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output_path and args.output_path.is_dir():
        output_dir = args.output_path
    elif args.output_path and len(urls) == 1:
        # Single URL with explicit file path - no default dir needed
        output_dir = None
    else:
        # Default to outputs/ directory
        output_dir = Path("outputs")

    # Create output directory if needed
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}", file=sys.stderr)

    # Handle dry-run mode
    if config.dry_run:
        print(f"Analyzing {len(urls)} URL(s) in dry-run mode...", file=sys.stderr)
        results = [dry_run_url(url, config) for url in urls]
        print_dry_run_report(results, config)
        return 0

    # Print configuration
    print(f"URLs to fetch: {len(urls)}", file=sys.stderr)
    print(f"Timeout: {config.timeout}s | Delay: {config.delay}s", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    processed = 0

    with GracefulShutdown() as shutdown:
        try:
            with BrowserSession(config) as session:
                converter = HTMLToMarkdown(config)

                for i, url in enumerate(urls):
                    if shutdown.shutdown_requested:
                        break

                    # Delay between requests (not before first)
                    if i > 0 and config.delay > 0:
                        if config.verbose:
                            print(
                                f"  [Waiting {config.delay}s...]",
                                file=sys.stderr,
                            )
                        time.sleep(config.delay)

                    try:
                        # Determine output path
                        # - Explicit path for single URL with -o
                        # - Auto-generate from title/URL otherwise
                        if len(urls) == 1 and args.output_path:
                            out_path = args.output_path
                            out_dir = None
                        else:
                            out_path = None
                            out_dir = output_dir

                        result = process_url(
                            url, session, converter, config, out_path, out_dir
                        )
                        if result is not None:
                            processed += 1

                    except (
                        requests.RequestException,
                        ValueError,
                        OSError,
                        UnicodeDecodeError,
                    ) as e:
                        print(f"Error fetching {url}: {e}", file=sys.stderr)
                        if config.verbose:
                            import traceback

                            traceback.print_exc()
                        continue

        except KeyboardInterrupt:
            print("\n[Interrupted]", file=sys.stderr)
        except (OSError, RuntimeError) as e:
            print(f"Fatal error: {e}", file=sys.stderr)
            if config.verbose:
                import traceback

                traceback.print_exc()
            return 1
        finally:
            gc.collect()

    print("-" * 70, file=sys.stderr)
    print(f"Processed: {processed}/{len(urls)} URLs", file=sys.stderr)

    return 0 if processed == len(urls) else 1


if __name__ == "__main__":
    sys.exit(main())
