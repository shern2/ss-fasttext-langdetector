import pytest

from ss_fasttext_langdetector import LangDetector


@pytest.fixture(scope="module")
def detector():
    """Fixture to create a LangDetector instance."""
    # This will download the model to the cache dir on the first run
    return LangDetector()


@pytest.mark.integration
def test_detect_english(detector):
    """Test detecting English text."""
    assert detector("This is a test in English.") == "en"


@pytest.mark.integration
def test_detect_french(detector):
    """Test detecting French text."""
    assert detector("Ceci est un test en français.") == "fr"


@pytest.mark.integration
def test_detect_batch(detector):
    """Test detecting batch of texts."""
    texts = ["Hello world", "Hola mundo"]
    assert detector.detect(texts) == ["en", "es"]


@pytest.mark.integration
def test_detect_multiline(detector):
    """Test detecting multiline text."""
    # Tests the paragraph voting logic
    text = "First paragraph in English.\n\nSecond paragraph also in English."
    assert detector(text) == "en"
