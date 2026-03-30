"""
Text Processing Module

Provides optional LLM-based post-processing of transcribed text to clean up
filler words, fix repetitions, and apply smart formatting.
"""

from typing import Any

from whisper_to_me.logger import get_logger

# Default system prompt for LLM text cleanup
DEFAULT_SYSTEM_PROMPT = """\
You are a speech-to-text post-processor. Your job is to clean up raw transcriptions \
while preserving the speaker's exact meaning.

## Rules

1. **Remove filler words**: Strip "um", "uh", "you know", "like" (when used as filler), \
"I mean", "sort of", "kind of" (when used as hedging), "basically", "actually" \
(when used as filler), "right", "so" (when used as filler at the start).

2. **Fix repetitions**: When the speaker restarts a sentence or repeats words, keep only \
the final, complete version. Example: "I want to I want to go to the store" → \
"I want to go to the store"

3. **Smart formatting**: Detect structural intent and format accordingly:
   - Explicit list cues ("make a list", "first... second...") → bullet list with "- " prefix
   - Numbered items ("number one", "number two") → numbered list ("1. ", "2. ")
   - "new paragraph" or "next paragraph" → insert a blank line
   - "heading" or "title" followed by text → "# " prefix (markdown heading)

4. **Punctuation and grammar**: Add proper punctuation, fix obvious grammar errors from \
speech recognition, capitalize sentence starts.

5. **Preserve meaning**: Never change WHAT the speaker said — only clean up HOW it's \
presented. Do not add information, rephrase ideas, or summarize.

6. **Output only the cleaned text**: No explanations, no commentary, no markdown code \
fences. Just the cleaned-up text, ready to be typed."""


class TextProcessor:
    """
    Optional LLM-based post-processor for transcribed text.

    Supports multiple backends (Ollama, OpenAI-compatible) and falls back
    to raw text on failure. Disabled by default (passthrough mode).
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
        """
        self.enabled = enabled
        self.backend = backend
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout = timeout
        self.logger = get_logger()

        if self.enabled:
            self.logger.info(
                f"Text processing enabled: {self.backend}/{self.model}",
                "processing",
            )

    def process(self, text: str) -> str:
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
            else:
                self.logger.warning(
                    f"Unknown processing backend '{self.backend}', using raw text",
                    "processing",
                )
                return text

            if result and result.strip():
                self.logger.debug(f"Processed text: '{result}'", "processing")
                return result.strip()
            else:
                self.logger.warning(
                    "LLM returned empty result, using raw text", "processing"
                )
                return text

        except Exception as e:
            self.logger.error(
                f"Text processing failed ({type(e).__name__}: {e}), using raw text",
                "processing",
            )
            return text

    def _process_ollama(self, text: str) -> str:
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
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            options={
                "temperature": self.temperature,
            },
        )

        return response.message.content

    def _process_openai(self, text: str) -> str:
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
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
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
