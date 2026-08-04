from types import SimpleNamespace
from unittest.mock import patch

from whisper_to_me.context_builder import ContextBuilder


def make_config(**overrides):
    defaults = {
        "enabled": True,
        "include_window_title": True,
        "base": "",
        "asr_prompt": "",
        "processing_prompt": "",
        "terms": [],
        "rolling_glossary_enabled": True,
        "rolling_glossary_reset_on_context_change": True,
        "rolling_glossary_max_terms": 120,
        "rolling_glossary_context_terms": 40,
        "max_asr_chars": 12000,
        "max_processing_chars": 12000,
        "rules": {},
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def focused_window(app="org.wezfurlong.wezterm", title="π - /home/marcos/pi-assistant"):
    return patch("whisper_to_me.display_backend.get_focused_window", return_value=(app, title))


def test_asr_context_puts_exact_spelling_glossary_early():
    config = make_config(
        terms=["Qwen ASR", "ROCm"],
        rules={
            "pi": {
                "match_title": ["π -"],
                "terms": ["Ollama", "AMD APU"],
                "hint": "User is talking to pi.",
            }
        },
    )
    builder = ContextBuilder(config)

    with focused_window():
        context = builder.build_asr_context()

    assert "Use these exact spellings when acoustically plausible:" in context
    glossary_index = context.index("Use these exact spellings")
    base_prompt_index = context.index("Use this context only")
    assert glossary_index < base_prompt_index
    assert glossary_index < 400
    assert "Qwen ASR" in context
    assert "ROCm" in context
    assert "Ollama" in context
    assert "AMD APU" in context


def test_rolling_glossary_learns_terms_for_next_utterance():
    config = make_config(terms=["Qwen ASR"])
    builder = ContextBuilder(config)

    with focused_window():
        builder.observe_text(
            "Use a separate LXC, test CPU first, then ROCm cautiously on the AMD APU."
        )
        context = builder.build_asr_context()

    assert "LXC" in builder.get_rolling_terms()
    assert "CPU" in builder.get_rolling_terms()
    assert "ROCm" in builder.get_rolling_terms()
    assert "AMD APU" in builder.get_rolling_terms()
    assert "Qwen ASR" in context
    assert "ROCm" in context
    assert "AMD APU" in context


def test_rolling_glossary_does_not_learn_simple_titlecase_misrecognitions():
    terms = ContextBuilder.extract_terms(
        "then Rockom cautiously and Olama because Quan ASI is bursty"
    )

    assert "Rockom" not in terms
    assert "Olama" not in terms
    assert "Quan ASI" not in terms
    # Acronyms may be useful, but the bad phrase should not be learned wholesale.
    assert "ASI" in terms


def test_rolling_glossary_does_not_join_terms_across_punctuation():
    terms = ContextBuilder.extract_terms("Use FooBarCLI, pyproject.toml, VCN, and VAAPI.")

    assert "FooBarCLI" in terms
    assert "pyproject.toml" in terms
    assert "VCN" in terms
    assert "VAAPI" in terms
    assert "FooBarCLI pyproject.toml" not in terms
    assert "pyproject.toml VCN" not in terms


def test_rolling_glossary_resets_when_focused_context_changes():
    config = make_config()
    builder = ContextBuilder(config)

    with focused_window(title="π - session one"):
        builder.observe_text("ROCm and VCN in session one")
        assert "ROCm" in builder.get_rolling_terms()

    with focused_window(title="π - session two"):
        context = builder.build_asr_context()

    assert "ROCm" not in builder.get_rolling_terms()
    assert "ROCm" not in context


def test_disabled_context_has_no_asr_context_or_learning():
    config = make_config(enabled=False, terms=["ROCm"])
    builder = ContextBuilder(config)

    with focused_window():
        builder.observe_text("VCN ROCm")
        assert builder.build_asr_context() == ""

    assert builder.get_rolling_terms() == []
