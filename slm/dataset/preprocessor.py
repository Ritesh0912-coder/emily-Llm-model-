"""
Text preprocessor for Emily SLM dataset pipeline.

Provides cleaning, deduplication, quality filtering, and normalisation
utilities for raw text corpora before tokenisation.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Apply a sequence of transforms to clean raw text before tokenisation.

    Example:
        >>> pp = TextPreprocessor(min_length=50, dedup=True)
        >>> clean_texts = pp.process(raw_texts)
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 100_000,
        dedup: bool = True,
        lowercase: bool = False,
        remove_urls: bool = True,
        normalise_unicode: bool = True,
        remove_control_chars: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.dedup = dedup
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.normalise_unicode = normalise_unicode
        self.remove_control_chars = remove_control_chars
        self._seen_hashes: set[str] = set()

    def process(self, texts: list[str]) -> list[str]:
        """
        Process a list of texts through the full pipeline.

        Args:
            texts: Raw text strings.

        Returns:
            Cleaned, deduplicated, filtered list of texts.
        """
        results: list[str] = []
        n_removed_short = 0
        n_removed_long = 0
        n_removed_dup = 0

        for text in texts:
            cleaned = self._clean(text)
            if len(cleaned) < self.min_length:
                n_removed_short += 1
                continue
            if len(cleaned) > self.max_length:
                n_removed_long += 1
                # Truncate rather than drop
                cleaned = cleaned[: self.max_length]

            if self.dedup:
                h = hashlib.md5(cleaned.encode()).hexdigest()
                if h in self._seen_hashes:
                    n_removed_dup += 1
                    continue
                self._seen_hashes.add(h)

            results.append(cleaned)

        logger.info(
            f"Preprocessing: {len(texts)} → {len(results)} texts kept | "
            f"removed: short={n_removed_short}, long={n_removed_long}, dup={n_removed_dup}"
        )
        return results

    def _clean(self, text: str) -> str:
        """Apply all enabled cleaning steps to a single text."""
        if self.normalise_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.remove_control_chars:
            # Keep \n and \t, remove other control characters
            text = "".join(
                ch for ch in text
                if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
            )

        if self.remove_urls:
            text = re.sub(r"https?://\S+", " ", text)
            text = re.sub(r"www\.\S+", " ", text)

        if self.lowercase:
            text = text.lower()

        # Normalise whitespace: collapse multiple spaces/tabs, strip
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def process_file(self, path: str | Path, output_path: str | Path) -> int:
        """
        Process a text file and write cleaned output.

        Args:
            path: Input file path.
            output_path: Output file path.

        Returns:
            Number of characters in the cleaned output.
        """
        path = Path(path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = path.read_text(encoding="utf-8", errors="replace")
        # Split into paragraphs for document-level dedup
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        cleaned = self.process(paragraphs)
        result = "\n\n".join(cleaned)
        output_path.write_text(result, encoding="utf-8")
        logger.info(f"Processed {path} → {output_path} ({len(result):,} chars)")
        return len(result)

    def reset_dedup(self) -> None:
        """Clear the deduplication hash set (start fresh)."""
        self._seen_hashes.clear()
