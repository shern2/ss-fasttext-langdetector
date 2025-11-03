"""
Contains the Language detector helper class for fasttext.
https://fasttext.cc/
"""

import logging
import re
from pathlib import Path
from urllib.request import urlretrieve

import fasttext

logger = logging.getLogger(__name__)
fasttext.FastText.eprint = lambda x: None  # Suppress FastText warnings


class LangDetector:
    """Language Detector helper class using FastText model."""

    def __init__(self, pth: Path | None = None, first_n_paras: int = 7):
        """
        Args:
            pth: Path to the FastText model.
                If `pth` is None, it will use the default model from the package.
                If `pth` is a Path object, it will be used as-is;
            first_n_paras (int): Number of '\n+' separated paragraphs to consider during 'voting' for language detection.
        """
        self.rgx_split_newline = re.compile(r"\n+")
        self.rgx_spaces = re.compile(r"\s+")
        self.first_n_paras = first_n_paras

        if pth is not None:
            pth_local = pth
        # case: Load model artifact from GitHub release
        else:
            pth_local = Path("/tmp/fasttext_lid.176.bin")
            urlretrieve(
                "https://github.com/shern2/ss-fasttext-langdetector/releases/download/v0.0.1/fasttext_lid.176.bin",
                pth_local,
            )
        self.model = fasttext.load_model(pth_local.as_posix())

    def __call__(self, text: str) -> str:
        """Detect the language of the `text`.
        Returns the corresponding language ISO code.

        Note: detects language per paragraph (Paragraphs are split by '\n+') and blank paragraphs are ignored.
        Takes the top-vote for the language of the first `self.first_n_paras` paragraphs.
        """
        paras = [
            para
            for para in (
                self.rgx_split_newline.split(
                    text.strip(),
                    maxsplit=self.first_n_paras + 1,  # +1 to avoid n-th para being a huge text chunk
                )[
                    # exclude the remaining paragraphs which may contain '\n' that FastText doesn't like
                    : self.first_n_paras
                ]
            )
            if not self.rgx_spaces.match(para)
        ]

        langs, _ = self.model.predict(paras)
        return max(langs, key=langs.count)[0][len("__label__") :]

    def detect(self, texts: list[str]) -> list[str]:
        """
        Given a list of `texts`, detects the language of each text.

        Args:
        `texts`: The list of texts to process.

        Returns:
        List of languages (2-letter ISO codes) corresponding to each text.
        """
        return [self(text) for text in texts]
