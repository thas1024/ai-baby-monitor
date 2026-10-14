from ai_baby_monitor import i18n


def test_default_language_is_chinese(monkeypatch):
    monkeypatch.delenv("APP_LANG", raising=False)
    assert i18n.get_language() == "zh"
    assert i18n.t("select_room") == "选择房间"


def test_english_translation(monkeypatch):
    monkeypatch.setenv("APP_LANG", "en")
    assert i18n.get_language() == "en"
    assert i18n.t("select_room") == "Select Room"


def test_unsupported_language_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("APP_LANG", "fr")
    assert i18n.get_language() == "zh"


def test_placeholders_are_filled():
    assert i18n.t("frame_width", lang="en", value=640) == "Frame width: 640"
    assert i18n.t("frame_width", lang="zh", value=640) == "画面宽度： 640"


def test_explicit_lang_overrides_env(monkeypatch):
    monkeypatch.setenv("APP_LANG", "en")
    assert i18n.t("mode_realtime", lang="zh") == "实时画面"


def test_missing_key_returns_key():
    assert i18n.t("does_not_exist", lang="en") == "does_not_exist"


def test_awareness_labels(monkeypatch):
    monkeypatch.setenv("APP_LANG", "zh")
    assert i18n.t("awareness_HIGH") == "高"
    monkeypatch.setenv("APP_LANG", "en")
    assert i18n.t("awareness_HIGH") == "HIGH"
