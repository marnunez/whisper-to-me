"""Test text processor functionality."""

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from whisper_to_me.text_processor import (
    DEFAULT_SYSTEM_PROMPT,
    TextProcessingError,
    TextProcessor,
)


class TestTextProcessorPassthrough:
    """Test passthrough behaviour when processing is disabled."""

    def test_disabled_by_default(self):
        """TextProcessor should be disabled by default."""
        processor = TextProcessor()
        assert processor.enabled is False

    def test_passthrough_when_disabled(self):
        """Disabled processor should return text unchanged."""
        processor = TextProcessor(enabled=False)
        assert processor.process("hello world") == "hello world"

    def test_passthrough_empty_text(self):
        """Should return empty text unchanged regardless of enabled state."""
        processor = TextProcessor(enabled=True)
        assert processor.process("") == ""
        assert processor.process("   ") == "   "

    def test_passthrough_none_text(self):
        """Should handle None gracefully."""
        processor = TextProcessor(enabled=True)
        assert processor.process(None) is None


class TestTextProcessorConfig:
    """Test configuration and initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        processor = TextProcessor()
        assert processor.backend == "ollama"
        assert processor.model == "qwen3:4b"
        assert processor.api_url == ""
        assert processor.api_key == ""
        assert processor.temperature == 0.3
        assert processor.timeout == 10
        assert processor.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_custom_config(self):
        """Test custom configuration."""
        processor = TextProcessor(
            enabled=True,
            backend="openai",
            model="gpt-4o-mini",
            api_url="https://api.openai.com/v1",
            api_key="sk-test",
            temperature=0.5,
            system_prompt="Custom prompt",
            timeout=30,
        )
        assert processor.enabled is True
        assert processor.backend == "openai"
        assert processor.model == "gpt-4o-mini"
        assert processor.api_url == "https://api.openai.com/v1"
        assert processor.api_key == "sk-test"
        assert processor.temperature == 0.5
        assert processor.system_prompt == "Custom prompt"
        assert processor.timeout == 30

    def test_empty_system_prompt_uses_default(self):
        """Empty system prompt should use the default."""
        processor = TextProcessor(system_prompt="")
        assert processor.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_custom_system_prompt(self):
        """Custom system prompt should override default."""
        processor = TextProcessor(system_prompt="Be brief.")
        assert processor.system_prompt == "Be brief."

    def test_get_info(self):
        """Test get_info returns correct structure."""
        processor = TextProcessor(
            enabled=True,
            backend="ollama",
            model="qwen3:4b",
            temperature=0.3,
        )
        info = processor.get_info()
        assert info["enabled"] is True
        assert info["backend"] == "ollama"
        assert info["model"] == "qwen3:4b"
        assert info["temperature"] == 0.3
        assert info["custom_prompt"] is False

    def test_get_info_custom_prompt(self):
        """get_info should report custom prompt usage."""
        processor = TextProcessor(system_prompt="Custom")
        info = processor.get_info()
        assert info["custom_prompt"] is True


def _make_ollama_mock(content="cleaned text"):
    """Create a mock ollama module with a working Client."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Client.return_value = mock_client
    mock_response = Mock()
    mock_response.message.content = content
    mock_client.chat.return_value = mock_response
    return mock_module, mock_client


def _make_openai_mock(content="cleaned text"):
    """Create a mock openai module with a working OpenAI client."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = content
    mock_client.chat.completions.create.return_value = mock_response
    return mock_module, mock_client


class TestTextProcessorOllama:
    """Test Ollama backend integration."""

    def test_ollama_success(self):
        """Test successful Ollama processing."""
        mock_module, mock_client = _make_ollama_mock("cleaned text")
        processor = TextProcessor(enabled=True, backend="ollama", model="qwen3:4b")

        with patch.dict(sys.modules, {"ollama": mock_module}):
            result = processor.process("um hello like world")

        assert result == "cleaned text"
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["model"] == "qwen3:4b"
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "/no_think\n[TRANSCRIPTION]\num hello like world\n[/TRANSCRIPTION]"

    def test_ollama_with_custom_url(self):
        """Test Ollama with custom API URL."""
        mock_module, _ = _make_ollama_mock("result")
        processor = TextProcessor(
            enabled=True,
            backend="ollama",
            api_url="http://localhost:11434",
        )

        with patch.dict(sys.modules, {"ollama": mock_module}):
            processor.process("test")

        mock_module.Client.assert_called_once_with(host="http://localhost:11434")

    def test_ollama_empty_response_raises(self):
        """Empty LLM response should raise, never fall back to raw text."""
        mock_module, _ = _make_ollama_mock("")
        processor = TextProcessor(enabled=True, backend="ollama")

        with patch.dict(sys.modules, {"ollama": mock_module}), \
             pytest.raises(TextProcessingError, match="empty result"):
            processor.process("original text")

    def test_ollama_exception_raises(self):
        """Ollama errors should raise, never fall back to raw text."""
        mock_module, mock_client = _make_ollama_mock()
        mock_client.chat.side_effect = ConnectionError("Ollama not running")
        processor = TextProcessor(enabled=True, backend="ollama")

        with patch.dict(sys.modules, {"ollama": mock_module}), \
             pytest.raises(TextProcessingError, match="Ollama not running"):
            processor.process("original text")

    def test_ollama_temperature_passed(self):
        """Test that temperature is passed to Ollama."""
        mock_module, mock_client = _make_ollama_mock("result")
        processor = TextProcessor(enabled=True, backend="ollama", temperature=0.7)

        with patch.dict(sys.modules, {"ollama": mock_module}):
            processor.process("test")

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.7


class TestTextProcessorOpenAI:
    """Test OpenAI backend integration."""

    def test_openai_success(self):
        """Test successful OpenAI processing."""
        mock_module, mock_client = _make_openai_mock("cleaned text")
        processor = TextProcessor(
            enabled=True,
            backend="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
        )

        with patch.dict(sys.modules, {"openai": mock_module}):
            result = processor.process("um hello like world")

        assert result == "cleaned text"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_openai_with_custom_url(self):
        """Test OpenAI with custom base URL (local server)."""
        mock_module, _ = _make_openai_mock("result")
        processor = TextProcessor(
            enabled=True,
            backend="openai",
            api_url="http://localhost:8080/v1",
        )

        with patch.dict(sys.modules, {"openai": mock_module}):
            processor.process("test")

        call_kwargs = mock_module.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:8080/v1"

    def test_openai_no_api_key_uses_dummy(self):
        """Test that missing API key uses 'not-needed' placeholder."""
        mock_module, _ = _make_openai_mock("result")
        processor = TextProcessor(enabled=True, backend="openai", api_key="")

        with patch.dict(sys.modules, {"openai": mock_module}):
            processor.process("test")

        call_kwargs = mock_module.OpenAI.call_args.kwargs
        assert call_kwargs["api_key"] == "not-needed"

    def test_openai_exception_raises(self):
        """OpenAI errors should raise, never fall back to raw text."""
        mock_module, mock_client = _make_openai_mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        processor = TextProcessor(enabled=True, backend="openai", api_key="sk-test")

        with patch.dict(sys.modules, {"openai": mock_module}), \
             pytest.raises(TextProcessingError, match="API error"):
            processor.process("original text")

    def test_openai_temperature_passed(self):
        """Test that temperature is passed to OpenAI."""
        mock_module, mock_client = _make_openai_mock("result")
        processor = TextProcessor(
            enabled=True, backend="openai", api_key="sk-test", temperature=0.5
        )

        with patch.dict(sys.modules, {"openai": mock_module}):
            processor.process("test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5


class TestTextProcessorOpenAICodex:
    """Test OpenAI Codex backend integration."""

    def test_resolve_openai_codex_url_default(self):
        """Empty URL should use the ChatGPT Codex endpoint."""
        assert TextProcessor._resolve_openai_codex_url("") == (
            "https://chatgpt.com/backend-api/codex/responses"
        )

    def test_resolve_openai_codex_url_base(self):
        """Base URL should resolve to /codex/responses."""
        assert TextProcessor._resolve_openai_codex_url("https://example.com/api") == (
            "https://example.com/api/codex/responses"
        )
        assert TextProcessor._resolve_openai_codex_url("https://example.com/api/codex") == (
            "https://example.com/api/codex/responses"
        )

    def test_parse_openai_codex_sse_delta(self):
        """SSE output_text deltas should be joined."""
        response = [
            b'data: {"type":"response.output_text.delta","delta":"cleaned"}\n',
            b"\n",
            b'data: {"type":"response.output_text.delta","delta":" text"}\n',
            b"\n",
        ]

        assert TextProcessor._parse_openai_codex_sse(response) == "cleaned text"

    def test_parse_openai_codex_sse_final_item(self):
        """Final message items should override delta text."""
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "final text"}],
            },
        }
        response = [f"data: {json.dumps(event)}\n".encode(), b"\n"]

        assert TextProcessor._parse_openai_codex_sse(response) == "final text"

    def test_openai_codex_success(self):
        """Test successful OpenAI Codex processing."""
        processor = TextProcessor(
            enabled=True,
            backend="openai-codex",
            model="gpt-5.5",
            temperature=0.4,
        )
        response = [
            b'data: {"type":"response.output_text.delta","delta":"cleaned text"}\n',
            b"\n",
        ]

        with patch.object(
            processor,
            "_get_openai_codex_credentials",
            return_value=("token", "account-id"),
        ), patch("whisper_to_me.text_processor.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = response
            result = processor.process("um hello like world")

        assert result == "cleaned text"
        request = mock_urlopen.call_args.args[0]
        body = json.loads(request.data.decode())
        assert body["model"] == "gpt-5.5"
        assert "temperature" not in body
        assert body["input"][0]["content"][0]["text"] == (
            "[TRANSCRIPTION]\num hello like world\n[/TRANSCRIPTION]"
        )
        assert request.get_header("Authorization") == "Bearer token"
        assert request.get_header("Chatgpt-account-id") == "account-id"


class TestTextProcessorUnknownBackend:
    """Test unknown backend handling."""

    def test_unknown_backend_raises(self):
        """Unknown backend should raise, never fall back to raw text."""
        processor = TextProcessor(enabled=True, backend="unknown_backend")

        with pytest.raises(TextProcessingError, match="Unknown processing backend"):
            processor.process("original text")


class TestDefaultSystemPrompt:
    """Test the default system prompt content."""

    def test_prompt_mentions_filler_words(self):
        """Default prompt should mention filler word removal."""
        assert "filler" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_repetitions(self):
        """Default prompt should mention repetition handling."""
        assert "repetition" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_formatting(self):
        """Default prompt should mention formatting."""
        assert "format" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_preserve_meaning(self):
        """Default prompt should emphasise preserving meaning."""
        assert "meaning" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_not_empty(self):
        """Default prompt should be substantial."""
        assert len(DEFAULT_SYSTEM_PROMPT) > 200
