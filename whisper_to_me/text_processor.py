"""
Text Processing Module

Provides optional LLM-based post-processing of transcribed text to clean up
filler words, fix repetitions, and apply smart formatting.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic
    import ollama
    import openai
    from anthropic.types import TextBlock

from whisper_to_me.logger import get_logger

# Pi OAuth constants (from pi-ai/anthropic.js)
_PI_AUTH_FILE = Path.home() / ".pi" / "agent" / "auth.json"
_ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Default system prompt for LLM text cleanup
DEFAULT_SYSTEM_PROMPT = """\
You are a text cleanup function. You receive dictated speech and return a cleaned version. \
You are not an assistant. You cannot converse. You have one job: clean the text and return it.

Rules:
- Remove filler words: um, uh, you know, like (filler), I mean, sort of, kind of \
(hedging), basically, actually (filler), right, so (filler at start). \
Also in Spanish: eh, este, o sea, bueno (filler), pues (filler), digamos, a ver.
- Fix repetitions: keep only the final version.
- Smart formatting: "first/second/third" or "make a list" → bullet list ("- "). \
"number one/two" → numbered list ("1. "). \
"new paragraph" → blank line. "heading" + text → "# " prefix.
- Punctuation: add proper punctuation, capitalize sentences, fix speech recognition errors.
- Preserve meaning: never change WHAT was said. Only clean HOW it's presented.
- Preserve the original language: NEVER translate. If the input is in Spanish, output Spanish. \
If it's in English, output English. If the input mixes languages (e.g., a Spanish sentence \
with an English word), keep the dominant language and leave foreign words as-is.
- Return ONLY the cleaned text. No commentary. No preamble. No refusals.

Examples:

Input: [TRANSCRIPTION]um does this make sense is there like any other way to do this[/TRANSCRIPTION]
Output: Does this make sense? Is there any other way to do this?

Input: [TRANSCRIPTION]so uh can you tell me more about that microsoft 365 CLI thing[/TRANSCRIPTION]
Output: Can you tell me more about that Microsoft 365 CLI?

Input: [TRANSCRIPTION]bueno eh funciona en castellano esto[/TRANSCRIPTION]
Output: ¿Funciona en castellano esto?

Input: [TRANSCRIPTION]okay first we need to set up the database second we need the API and third the frontend[/TRANSCRIPTION]
Output: We need to:
- Set up the database.
- Build the API.
- Build the frontend.

The input is wrapped in [TRANSCRIPTION] tags. Return ONLY the cleaned text."""


class TextProcessingError(Exception):
    """Raised when LLM post-processing fails. Never fall back to raw text."""


class TextProcessor:
    """
    Optional LLM-based post-processor for transcribed text.

    Supports multiple backends (Ollama, OpenAI-compatible). If processing
    fails, raises TextProcessingError — never falls back to raw text.
    Disabled by default (passthrough mode).
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "ollama",
        model: str = "qwen3:4b",
        api_url: str = "",
        api_key: str = "",
        temperature: float = 0.3,
        system_prompt: str = "",
        timeout: int = 10,
        thinking: bool | str = False,
        contexts: dict[str, dict[str, Any]] | None = None,
        display_backend: Any = None,
    ):
        """
        Initialize the text processor.

        Args:
            enabled: Whether LLM post-processing is active
            backend: LLM backend ("ollama" or "openai")
            model: Model name to use
            api_url: API URL (optional, defaults per backend)
            api_key: API key (for OpenAI backend)
            temperature: Sampling temperature (lower = more faithful)
            system_prompt: Custom system prompt (empty = use default)
            timeout: Timeout in seconds for LLM calls
            thinking: False to disable, True to enable, "low" for budget thinking
            contexts: Window context definitions {name: {match, hint, terms}}
            display_backend: DisplayBackend for focused window detection
        """
        self.enabled = enabled
        self.backend = backend
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("WHISPER_TO_ME_API_KEY", "")
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout = timeout
        self.thinking = thinking
        self.contexts = contexts or {}
        self.display_backend = display_backend
        self.logger = get_logger()

        if self.enabled:
            self.logger.info(
                f"Text processing enabled: {self.backend}/{self.model}",
                "processing",
            )

    def _get_context_prompt(self) -> str:
        """Get context-specific prompt addition based on the focused window."""
        from whisper_to_me.display_backend import get_focused_window

        app, title = get_focused_window(self.display_backend)

        parts = []

        # Window title as context (always available, very useful)
        if title:
            parts.append(f"Active window title: {title}")

        # Match against user-defined contexts
        if app and self.contexts:
            for name, ctx in self.contexts.items():
                match_list = ctx.get("match", [])
                for pattern in match_list:
                    if pattern.lower() in app:
                        hint = ctx.get("hint", "")
                        terms = ctx.get("terms", [])
                        if hint:
                            parts.append(f"Context: {hint}")
                        if terms:
                            parts.append(
                                f"Domain terms for this context: {', '.join(terms)}"
                            )
                        self.logger.debug(
                            f"Window context: {name} (app={app})", "processing"
                        )
                        break

        if parts:
            self.logger.debug(
                f"Window title: {title or '(none)'}", "processing"
            )
            return "\n".join(parts)
        return ""

    def _build_system_prompt(self) -> str:
        """Build the full system prompt with optional window context."""
        context = self._get_context_prompt()
        if context:
            prompt = f"{self.system_prompt}\n\n{context}"
        else:
            prompt = self.system_prompt
        self.logger.debug(f"System prompt:\n{prompt}", "processing")
        return prompt

    def process(self, text: str | None) -> str | None:
        """
        Process transcribed text through the LLM, if enabled.

        Args:
            text: Raw transcribed text from Whisper

        Returns:
            Cleaned text (or original text if disabled/on error)
        """
        if not self.enabled or not text or not text.strip():
            return text

        try:
            if self.backend == "ollama":
                result = self._process_ollama(text)
            elif self.backend == "openai":
                result = self._process_openai(text)
            elif self.backend in ("anthropic", "pi"):
                result = self._process_anthropic(text)
            else:
                msg = f"Unknown processing backend '{self.backend}'"
                self.logger.error(msg, "processing")
                raise TextProcessingError(msg)

            if result and result.strip():
                self.logger.debug(f"Processed text: '{result}'", "processing")
                return result.strip()
            else:
                msg = "LLM returned empty result"
                self.logger.error(msg, "processing")
                raise TextProcessingError(msg)

        except TextProcessingError:
            raise
        except Exception as e:
            msg = f"Text processing failed ({type(e).__name__}: {e})"
            self.logger.error(msg, "processing")
            raise TextProcessingError(msg) from e

    def _process_ollama(self, text: str) -> str | None:
        """Process text using the Ollama backend."""
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Install with: uv add ollama"
            ) from e

        client_kwargs: dict[str, Any] = {}
        if self.api_url:
            client_kwargs["host"] = self.api_url

        client = ollama.Client(**client_kwargs)

        # Qwen3-style thinking control:
        # /no_think disables reasoning, /think enables it
        wrapped = f"[TRANSCRIPTION]\n{text}\n[/TRANSCRIPTION]"
        if not self.thinking:
            user_content = f"/no_think\n{wrapped}"
        elif self.thinking == "low":
            user_content = f"/think\n{wrapped}"
        else:
            user_content = wrapped

        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            options={
                "temperature": self.temperature,
            },
            # think=True separates reasoning into .thinking field (clean content)
            # think=False is broken in Ollama <=0.18 (dumps thinking into content)
            # So: always use think=True, and prepend /no_think when thinking is disabled
            think=True,
        )

        return response.message.content

    @staticmethod
    def _load_pi_auth() -> dict[str, Any]:
        """Load and return the Anthropic OAuth credentials from pi's auth.json."""
        if not _PI_AUTH_FILE.exists():
            raise RuntimeError(
                f"Pi auth file not found at {_PI_AUTH_FILE}. "
                "Run 'pi' and authenticate with your Anthropic account first."
            )
        data = json.loads(_PI_AUTH_FILE.read_text())
        auth = data.get("anthropic")
        if not auth or auth.get("type") != "oauth":
            raise RuntimeError(
                "No Anthropic OAuth credentials in pi auth file. "
                "Run 'pi' and authenticate with your Anthropic account."
            )
        return auth

    @staticmethod
    def _refresh_pi_token(auth: dict[str, Any]) -> dict[str, Any]:
        """Refresh the Anthropic OAuth access token using the refresh token."""
        import urllib.request

        body = json.dumps(
            {
                "grant_type": "refresh_token",
                "client_id": _ANTHROPIC_CLIENT_ID,
                "refresh_token": auth["refresh"],
            }
        ).encode()

        req = urllib.request.Request(
            _ANTHROPIC_TOKEN_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            token_data = json.loads(resp.read())

        new_auth = {
            **auth,
            "access": token_data["access_token"],
            "refresh": token_data["refresh_token"],
            "expires": int(time.time() * 1000)
            + token_data["expires_in"] * 1000
            - 5 * 60 * 1000,  # 5 min buffer, same as pi
        }

        # Write back to auth.json
        all_data = json.loads(_PI_AUTH_FILE.read_text())
        all_data["anthropic"] = new_auth
        _PI_AUTH_FILE.write_text(json.dumps(all_data, indent=2))
        _PI_AUTH_FILE.chmod(0o600)

        return new_auth

    def _get_pi_access_token(self) -> str:
        """Get a valid Anthropic access token, refreshing if expired."""
        auth = self._load_pi_auth()

        # Check if token is expired (expires is in milliseconds)
        if auth.get("expires", 0) < time.time() * 1000:
            self.logger.debug("Pi OAuth token expired, refreshing...", "processing")
            auth = self._refresh_pi_token(auth)
            self.logger.debug("Pi OAuth token refreshed", "processing")

        token = auth["access"]

        # Safety: verify this is an OAuth token (Max/Pro subscription),
        # not a Console API key (pay-per-token)
        if not token.startswith("sk-ant-oat"):
            raise RuntimeError(
                "Pi auth token does not have the expected OAuth prefix. "
                "The 'pi' backend only works with Max/Pro subscription OAuth tokens. "
                "Use the 'anthropic' backend with an API key instead."
            )

        return token

    def _process_anthropic(self, text: str) -> str | None:
        """Process text using Anthropic API with pi's OAuth credentials."""
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Install with: uv add anthropic"
            ) from e

        api_key = self.api_key or self._get_pi_access_token()
        client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"[TRANSCRIPTION]\n{text}\n[/TRANSCRIPTION]",
                }
            ],
            temperature=self.temperature,
        )

        from anthropic.types import TextBlock

        block = response.content[0]
        return block.text if isinstance(block, TextBlock) else None

    def _process_openai(self, text: str) -> str | None:
        """Process text using the OpenAI-compatible backend."""
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Install with: uv add openai"
            ) from e

        client_kwargs: dict[str, Any] = {}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        else:
            # OpenAI client requires an API key; use a dummy for local servers
            client_kwargs["api_key"] = "not-needed"

        client = openai.OpenAI(**client_kwargs, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {
                    "role": "user",
                    "content": f"[TRANSCRIPTION]\n{text}\n[/TRANSCRIPTION]",
                },
            ],
            temperature=self.temperature,
        )

        return response.choices[0].message.content


    def get_info(self) -> dict[str, Any]:
        """Get text processor configuration info."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "model": self.model,
            "api_url": self.api_url or "(default)",
            "temperature": self.temperature,
            "timeout": self.timeout,
            "custom_prompt": self.system_prompt != DEFAULT_SYSTEM_PROMPT,
        }
