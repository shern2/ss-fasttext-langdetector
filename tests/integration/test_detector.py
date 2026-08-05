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


@pytest.mark.integration
def test_detect_tied_paragraph_votes_use_mean_confidence(detector):
    """A weak headline must not beat a highly confident body paragraph on a tie."""
    text = (
        "谷歌被罚！ - thepaper.cn\n\n"
        "欧盟委员会宣布，谷歌在网络搜索和应用商店业务中违反数字市场法，"
        "决定对其处以罚款。委员会表示，企业在搜索结果中优先展示自家购物、"
        "酒店、交通和体育服务，损害了其他竞争者和消费者的选择。"
    )
    assert detector(text) == "zh"
