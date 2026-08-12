import re
import tracemalloc

import pytest

from ss_fasttext_langdetector import LangDetector, _iter_paragraphs


@pytest.fixture(scope="module")
def detector():
    """Fixture to create a LangDetector instance."""
    # This will download the model to the cache dir on the first run
    return LangDetector()


def test_iter_paragraphs_is_lazy():
    """Paragraph iteration must not eagerly materialize an entire document."""
    text = "first\n\nsecond\n" + "trailing\n" * 200_000
    tracemalloc.start()
    try:
        paragraphs = _iter_paragraphs(text)
        assert iter(paragraphs) is paragraphs
        assert next(paragraphs) == "first"
        assert next(paragraphs) == "second"
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 1_000_000


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
@pytest.mark.parametrize("text", ["", "\n\n", "   \n\t  "])
def test_detect_empty_or_whitespace_only_input(detector, text):
    """Empty input should preserve FastText's existing fallback behavior without crashing."""
    assert detector(text) == "en"


@pytest.mark.integration
def test_detect_rejects_non_positive_paragraph_limit():
    """A non-positive paragraph limit should fail immediately with a clear error."""
    with pytest.raises(ValueError, match="first_n_paras must be positive"):
        LangDetector(first_n_paras=0)


@pytest.mark.integration
def test_detect_multiline(detector):
    """Test detecting multiline text."""
    # Tests the paragraph voting logic
    text = "First paragraph in English.\n\nSecond paragraph also in English."
    assert detector(text) == "en"


@pytest.mark.integration
def test_detect_weighted_votes_favor_confident_body(detector):
    """A weak headline must not beat a highly confident body paragraph."""
    text = (
        "谷歌被罚！ - thepaper.cn\n\n"
        "欧盟委员会宣布，谷歌在网络搜索和应用商店业务中违反数字市场法，"
        "决定对其处以罚款。委员会表示，企业在搜索结果中优先展示自家购物、"
        "酒店、交通和体育服务，损害了其他竞争者和消费者的选择。"
    )
    assert detector(text) == "zh"


@pytest.mark.integration
def test_detect_keeps_short_non_metadata_paragraphs(detector):
    """Short conversational paragraphs remain votes even beside a longer paragraph."""
    text = (
        "Xin chào bạn.\n\n"
        "Cảm ơn nhiều.\n\n"
        "This deliberately longer English paragraph is present but must not erase two valid Vietnamese paragraphs."
    )
    assert detector(text) == "vi"


@pytest.mark.integration
def test_detect_keeps_short_colon_terminated_content(detector):
    """Punctuation alone must not make valid short paragraphs disappear."""
    text = (
        "Je vous le dis :\n\n"
        "Nous sommes là :\n\n"
        "This deliberately longer English advertisement must not erase two valid French paragraphs."
    )
    assert detector(text) == "fr"


@pytest.mark.integration
def test_detect_majority_language_survives_long_foreign_ads(detector):
    """A minority of long foreign-language ads must not override the paragraph majority."""
    vietnamese_news = [
        "Đây là tin mới.",
        "Thông tin hôm nay.",
        "Nội dung chính.",
        "Cập nhật thị trường.",
    ]
    english_ads = [
        "This advertisement promotes international travel, hotels, discount flights, and shopping offers.",
        "Visit our website to discover promotions, commercial products, subscriptions, and sponsored services.",
        "This paid promotional message comes from an external partner and is not part of the original report.",
    ]
    assert detector("\n\n".join(vietnamese_news + english_ads)) == "vi"


@pytest.mark.integration
def test_detect_majority_language_survives_long_foreign_ads_in_reverse(detector):
    """The ad-resistance behavior must work in both language directions."""
    english_news = [
        "This is the latest report.",
        "The main story follows.",
        "We confirm the update.",
        "More market news today.",
    ]
    vietnamese_ads = [
        "Quảng cáo này giới thiệu dịch vụ du lịch, khách sạn, vé máy bay và nhiều ưu đãi mua sắm.",
        "Hãy truy cập trang web để khám phá chương trình khuyến mãi, sản phẩm và dịch vụ tài trợ.",
        "Thông điệp quảng cáo trả phí này đến từ đối tác bên ngoài và không thuộc về bản tin gốc.",
    ]
    assert detector("\n\n".join(english_news + vietnamese_ads)) == "en"


@pytest.mark.integration
def test_detect_bounds_non_alphabetic_paragraph_scanning(detector):
    """The configured limit must also bound scanning through numeric/symbol-only input."""
    text = "\n".join(["123456"] * 100 + ["This English paragraph is beyond the scan window."])
    assert detector(text) != "en"


@pytest.mark.integration
def test_detect_finds_language_after_numeric_prefix_within_scan_window(detector):
    """Generic numeric noise can be skipped when language-bearing text is still within the bound."""
    text = "\n".join(["123456"] * 20 + ["This is a clear English news paragraph."])
    assert detector(text) == "en"


@pytest.mark.integration
def test_detect_respects_first_n_language_bearing_paragraphs():
    """Later language-bearing text must not change a filled configured window."""
    text = "This is English.\nCeci est français.\nCeci est encore français."
    assert LangDetector(first_n_paras=1)(text) == "en"


@pytest.mark.integration
def test_detect_honors_custom_paragraph_separator(detector, monkeypatch):
    """Existing custom separator behavior remains effective after lazy iteration."""
    predicted_paragraphs = []
    original_predict = detector.model.predict

    def capture_predict(paragraphs, *args, **kwargs):
        predicted_paragraphs.extend(paragraphs)
        return original_predict(paragraphs, *args, **kwargs)

    monkeypatch.setattr(detector.model, "predict", capture_predict)
    detector.rgx_split_newline = re.compile(r"\|")
    try:
        detector("This is English.|Ceci est français.")
    finally:
        detector.rgx_split_newline = re.compile(r"\n+")
    assert predicted_paragraphs == ["This is English.", "Ceci est français."]


@pytest.mark.integration
def test_detect_honors_custom_whitespace_filter(detector, monkeypatch):
    """Existing custom paragraph-filter behavior remains effective."""
    predicted_paragraphs = []
    original_predict = detector.model.predict

    def capture_predict(paragraphs, *args, **kwargs):
        predicted_paragraphs.extend(paragraphs)
        return original_predict(paragraphs, *args, **kwargs)

    monkeypatch.setattr(detector.model, "predict", capture_predict)
    detector.rgx_spaces = re.compile(r"SKIP")
    try:
        detector("SKIP English text.\nCeci est français.")
    finally:
        detector.rgx_spaces = re.compile(r"\s+")
    assert predicted_paragraphs == ["Ceci est français."]


@pytest.mark.integration
def test_detect_unicode_scripts_count_as_language_bearing(detector):
    """Alphabetic scripts outside ASCII must not be mistaken for numeric/punctuation noise."""
    assert detector("这是一个清楚的中文新闻段落。") == "zh"


@pytest.mark.integration
def test_detect_short_table_cells_do_not_outvote_body():
    """Many tiny table cells must not outvote substantive Vietnamese paragraphs."""
    table_cells = [f"{index}." for index in range(1, 31)]
    body = ["Đây là một bài báo tiếng Việt về phát triển bền vững và kinh tế."] * 3
    assert LangDetector(first_n_paras=2000)("\n\n".join(table_cells + body)) == "vi"
