"""
Contains the Language detector helper class for fasttext.
https://fasttext.cc/
"""

import hashlib
import logging
import re
import sys
from collections.abc import Iterator
from itertools import islice
from math import log1p
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import urlretrieve

# When installed from a pre-built wheel, fasttext is vendored inside the package.
# Add the _vendor directory to sys.path so `import fasttext` resolves to the bundled copy.
_vendor_dir = Path(__file__).parent / "_vendor"
if _vendor_dir.exists() and str(_vendor_dir) not in sys.path:
    sys.path.insert(0, str(_vendor_dir))

import fasttext
from platformdirs import user_cache_dir
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
fasttext.FastText.eprint = lambda x: None  # Suppress FastText warnings

MODEL_VERSION = "v0.0.1"
MODEL_FILENAME = "fasttext_lid.176.bin"
MODEL_URL = f"https://github.com/shern2/ss-fasttext-langdetector/releases/download/{MODEL_VERSION}/{MODEL_FILENAME}"
MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
PARAGRAPH_SEPARATOR = re.compile(r"\n+")
SPACES = re.compile(r"\s+")
MAX_LENGTH_WEIGHT_CHARS = 20
SCAN_LIMIT_MULTIPLIER = 10


def _verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify file integrity using SHA256 hash."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def _iter_paragraphs(text: str, separator_pattern: re.Pattern[str] = PARAGRAPH_SEPARATOR) -> Iterator[str]:
    """Yield newline-separated paragraphs without materializing the full split."""
    start = 0
    for separator in separator_pattern.finditer(text):
        yield text[start : separator.start()]
        start = separator.end()
    yield text[start:]


@retry(
    retry=retry_if_exception_type((URLError, HTTPError, ContentTooShortError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _download_model(url: str, file_path: Path) -> None:
    """Download model file with retries."""
    urlretrieve(url, file_path)


class LangDetector:
    """Language Detector helper class using FastText model."""

    def __init__(self, pth: Path | None = None, first_n_paras: int = 7):
        """
        Args:
            pth: Path to the FastText model.
                If `pth` is None, it will use the default model from the package.
                If `pth` is a Path object, it will be used as-is;
            first_n_paras (int): Number of language-bearing '\n+' separated paragraphs to consider during voting.
        """
        if first_n_paras <= 0:
            raise ValueError("first_n_paras must be positive")
        # Kept as instance attributes for compatibility with callers that customize them.
        self.rgx_split_newline = PARAGRAPH_SEPARATOR
        self.rgx_spaces = SPACES
        self.first_n_paras = first_n_paras

        if pth is not None:
            pth_local = pth
        # case: Load model artifact from GitHub release
        else:
            cache_dir = Path(user_cache_dir("ss-fasttext-langdetector")) / MODEL_VERSION
            cache_dir.mkdir(parents=True, exist_ok=True)
            pth_local = cache_dir / MODEL_FILENAME

            if not pth_local.exists():
                logger.info("Downloading model from %s", MODEL_URL)
                try:
                    _download_model(MODEL_URL, pth_local)
                    if not _verify_file_hash(pth_local, MODEL_SHA256):
                        pth_local.unlink()  # Remove corrupted file
                        raise ValueError("Downloaded model failed integrity verification")
                    logger.info("Model cached at %s", pth_local)
                except Exception as e:
                    logger.error("Failed to download model: %s", e)
                    raise
            else:
                logger.debug("Using cached model at %s", pth_local)

        self.model = fasttext.load_model(pth_local.as_posix())

    def __call__(self, text: str) -> str:
        """Detect the language of the `text`.
        Returns the corresponding language ISO code.

        Note: detects language per paragraph (Paragraphs are split by '\n+') and blank paragraphs are ignored.
        Votes from the first `self.first_n_paras` paragraphs are weighted by confidence and
        capped log-scaled character length so tiny fragments have less influence while limiting
        the leverage of very long paragraphs.
        Non-alphabetic paragraphs are ignored when language-bearing content is available.
        """
        paras: list[str] = []
        fallback_paras: list[str] = []
        scan_limit = self.first_n_paras * SCAN_LIMIT_MULTIPLIER
        for raw_para in islice(_iter_paragraphs(text.strip(), self.rgx_split_newline), scan_limit):
            para = raw_para.strip()
            if not para or self.rgx_spaces.match(para):
                continue
            if len(fallback_paras) < self.first_n_paras:
                fallback_paras.append(para)
            if not any(char.isalpha() for char in para):
                continue
            paras.append(para)
            if len(paras) >= self.first_n_paras:
                break

        if not paras:
            paras = fallback_paras

        if not paras:
            paras = [""]

        labels, confidences = self.model.predict(paras, k=1)
        weighted_votes: dict[str, float] = {}
        for para, labels_for_para, confidences_for_para in zip(paras, labels, confidences, strict=True):
            label = labels_for_para[0]
            weighted_votes[label] = weighted_votes.get(label, 0.0) + float(confidences_for_para[0]) * log1p(
                min(len(para.strip()), MAX_LENGTH_WEIGHT_CHARS)
            )

        return str(max(weighted_votes, key=weighted_votes.__getitem__)[len("__label__") :])

    def detect(self, texts: list[str]) -> list[str]:
        """
        Given a list of `texts`, detects the language of each text.

        Args:
        `texts`: The list of texts to process.

        Returns:
        List of languages (2-letter ISO codes) corresponding to each text.
        """
        return [self(text) for text in texts]
