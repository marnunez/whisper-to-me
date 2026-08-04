"""Context construction for ASR and post-processing.

A single context registry can feed two different consumers:
- ASR context: recognition bias for models that support large text context.
- Processing context: semantic guidance for the LLM cleanup pass.

The two renderings intentionally differ. ASR context must be conservative and
lexical; post-processing context can include richer cleanup guidance.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from whisper_to_me.logger import get_logger

ContextKind = Literal["asr", "processing"]

DEFAULT_ASR_PROMPT = """\
Use this context only to improve speech recognition. Do not add words that were not spoken. \
Prefer exact spelling for commands, file names, package names, hostnames, APIs, and code identifiers \
when they are acoustically plausible. Preserve the speaker's language; never translate.\
"""

_GLOSSARY_PREFIX = "Use these exact spellings when acoustically plausible: "

# Conservative technical-term extraction. This intentionally avoids learning
# arbitrary Title Case words from imperfect transcripts: a mistaken "Olama" is
# much more damaging than omitting a novel simple word for one utterance.
_TERM_RE = re.compile(
    r"(?<![\w/.-])"
    r"("
    r"[A-Z]{2,}[a-z]*"  # ASR, API, ROCm-ish acronym tokens
    r"|[A-Za-z0-9]+(?:[-_/.:][A-Za-z0-9]+)+"  # qwen-asr, pyproject.toml, pbs.lan
    r"|[A-Za-z]*\d[A-Za-z0-9]*(?:[-_/.][A-Za-z0-9]+)*"  # Qwen3, large-v3
    r"|[A-Za-z]+[A-Z][A-Za-z0-9]*"  # TrueNAS, OpenAI, fastWhisper
    r")"
    r"(?![\w/-])"
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_/.:][A-Za-z0-9]+)*")

_STOPWORDS = {
    "A", "An", "And", "Are", "As", "At", "Be", "But", "By", "Can", "Do",
    "For", "From", "Has", "Have", "How", "If", "In", "Into", "Is", "It",
    "Its", "Let", "No", "Not", "Of", "On", "Or", "So", "That", "The",
    "Then", "There", "This", "To", "Use", "We", "What", "When", "Where",
    "With", "Would", "Yes", "You",
}


@dataclass(frozen=True)
class ActiveContext:
    """Resolved context for the currently focused window."""

    app: str
    title: str
    name: str | None
    rule: dict[str, Any]


@dataclass
class RollingTerm:
    """A term learned during the current focused context."""

    text: str
    score: float
    last_seen: float


class ContextBuilder:
    """Build ASR and post-processing context from a shared context registry."""

    def __init__(self, config: Any, display_backend: Any = None):
        self.config = config
        self.display_backend = display_backend
        self.logger = get_logger()
        self._rolling_terms: dict[str, RollingTerm] = {}
        self._context_signature: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    def build_asr_context(self) -> str:
        """Return conservative context for speech recognition backends."""
        if not self.enabled:
            return ""
        return self._build("asr", max_chars=getattr(self.config, "max_asr_chars", 12000))

    def build_processing_context(self) -> str:
        """Return richer context for LLM post-processing."""
        if not self.enabled:
            return ""
        return self._build(
            "processing",
            max_chars=getattr(self.config, "max_processing_chars", 12000),
        )

    def observe_text(self, text: str | None, weight: float = 1.0) -> None:
        """Learn technical terms from a successful dictation.

        The glossary is intentionally runtime-local. It helps the next utterance
        in the same focused context, but it is not persisted as long-term memory.
        """
        if not self.enabled or not getattr(self.config, "rolling_glossary_enabled", True):
            return
        if not text or not text.strip():
            return

        active = self._resolve_active_context()
        self._maybe_reset_for_active_context(active)
        self._observe_terms(self.extract_terms(text), weight=weight)

    def reset_rolling_glossary(self) -> None:
        """Forget runtime-learned terms."""
        self._rolling_terms.clear()

    def get_rolling_terms(self) -> list[str]:
        """Return learned terms ordered by usefulness."""
        return [term.text for term in self._ranked_rolling_terms()]

    def _build(self, kind: ContextKind, max_chars: int) -> str:
        active = self._resolve_active_context()
        self._maybe_reset_for_active_context(active)
        parts: list[str] = []

        if kind == "asr":
            glossary = self._build_glossary_line(active)
            if glossary:
                # Keep the exact-spelling glossary first. Qwen-ASR contexts may
                # be tightly capped by the caller, and this is the high-signal bit.
                parts.append(glossary)
            parts.append(getattr(self.config, "asr_prompt", "") or DEFAULT_ASR_PROMPT)
        elif processing_prompt := getattr(self.config, "processing_prompt", ""):
            parts.append(processing_prompt)

        if base := getattr(self.config, "base", ""):
            parts.append(base)

        if getattr(self.config, "include_window_title", True) and active:
            if active.title:
                parts.append(f"Active window title: {active.title}")
            if active.app:
                parts.append(f"Active application: {active.app}")

        if active and active.rule:
            if active.name:
                parts.append(f"Matched context: {active.name}")

            # Shared semantic hint.
            if hint := active.rule.get("hint", ""):
                parts.append(f"Context: {hint}")

            # Kind-specific prose. These are intentionally separate from the
            # shared hint: ASR should bias recognition; processing can clean up.
            specific = active.rule.get(kind, "")
            if specific:
                label = "ASR guidance" if kind == "asr" else "Processing guidance"
                parts.append(f"{label}: {specific}")

            terms = self._coerce_list(active.rule.get("terms", []))
            kind_terms = self._coerce_list(active.rule.get(f"{kind}_terms", []))
            all_terms = self._dedupe([*terms, *kind_terms])
            if all_terms:
                label = "Recognition terms" if kind == "asr" else "Domain terms"
                parts.append(f"{label}: {', '.join(all_terms)}")

            examples = self._coerce_list(active.rule.get("examples", []))
            kind_examples = self._coerce_list(active.rule.get(f"{kind}_examples", []))
            all_examples = self._dedupe([*examples, *kind_examples])
            if all_examples:
                parts.append("Examples:\n" + "\n".join(f"- {item}" for item in all_examples))

        context = "\n".join(part.strip() for part in parts if str(part).strip()).strip()
        return self._limit_context(context, max_chars)

    def _build_glossary_line(self, active: ActiveContext | None) -> str:
        terms = self._glossary_terms(active)
        if not terms:
            return ""
        return _GLOSSARY_PREFIX + ", ".join(terms) + "."

    def _glossary_terms(self, active: ActiveContext | None) -> list[str]:
        configured_terms = self._coerce_list(getattr(self.config, "terms", []))
        active_terms: list[str] = []
        title_terms: list[str] = []

        if active and active.rule:
            active_terms.extend(self._coerce_list(active.rule.get("terms", [])))
            active_terms.extend(self._coerce_list(active.rule.get("asr_terms", [])))
        if active and active.title:
            title_terms.extend(self.extract_terms(active.title))

        rolling_limit = int(getattr(self.config, "rolling_glossary_context_terms", 40))
        rolling_terms = [term.text for term in self._ranked_rolling_terms()[:rolling_limit]]
        return self._dedupe([*configured_terms, *rolling_terms, *active_terms, *title_terms])

    def _observe_terms(self, terms: list[str], weight: float = 1.0) -> None:
        now = time.time()
        for term in terms:
            key = term.casefold()
            existing = self._rolling_terms.get(key)
            if existing:
                existing.score += weight
                existing.last_seen = now
                # Keep the newest spelling/capitalisation.
                existing.text = term
            else:
                self._rolling_terms[key] = RollingTerm(text=term, score=weight, last_seen=now)

        max_terms = int(getattr(self.config, "rolling_glossary_max_terms", 120))
        if max_terms > 0 and len(self._rolling_terms) > max_terms:
            keep = {term.text.casefold() for term in self._ranked_rolling_terms()[:max_terms]}
            self._rolling_terms = {
                key: term for key, term in self._rolling_terms.items() if key in keep
            }

    def _ranked_rolling_terms(self) -> list[RollingTerm]:
        return sorted(
            self._rolling_terms.values(),
            key=lambda term: (term.score, term.last_seen, len(term.text)),
            reverse=True,
        )

    def _maybe_reset_for_active_context(self, active: ActiveContext | None) -> None:
        if not getattr(self.config, "rolling_glossary_reset_on_context_change", True):
            return
        signature = self._active_signature(active)
        if self._context_signature is None:
            self._context_signature = signature
            return
        if signature != self._context_signature:
            self.logger.debug("Resetting rolling glossary after context change", "context")
            self._context_signature = signature
            self.reset_rolling_glossary()

    @staticmethod
    def _active_signature(active: ActiveContext | None) -> str:
        if not active:
            return ""
        return "\0".join([active.name or "", active.app or "", active.title or ""])

    def _resolve_active_context(self) -> ActiveContext | None:
        try:
            from whisper_to_me.display_backend import get_focused_window

            app, title = get_focused_window(self.display_backend)
        except Exception as e:
            self.logger.debug(f"Could not read focused window for context: {e}", "context")
            app, title = "", ""

        rules = getattr(self.config, "rules", {}) or {}
        matched = self._match_context(app or "", rules, field="match_app")
        if matched is None:
            matched = self._match_context(app or "", rules, field="match")

        # Allow a broad app match to be refined by title-specific contexts.
        if matched is not None:
            _name, rule = matched
            if rule.get("check_title") and title:
                title_match = self._match_context(title, rules, field="match_title")
                if title_match:
                    matched = title_match
        elif title:
            matched = self._match_context(title, rules, field="match_title")

        name = matched[0] if matched else None
        rule = matched[1] if matched else {}
        if name:
            self.logger.debug(
                f"Window context: {name} (app={app or ''}, title={title or ''!r})",
                "context",
            )
        return ActiveContext(app=app or "", title=title or "", name=name, rule=rule)

    def _match_context(
        self,
        target: str,
        rules: dict[str, dict[str, Any]],
        field: str,
    ) -> tuple[str, dict[str, Any]] | None:
        if not target:
            return None
        target_lower = target.lower()
        for name, rule in rules.items():
            for pattern in self._coerce_list(rule.get(field, [])):
                if str(pattern).lower() in target_lower:
                    return name, rule
        return None

    @staticmethod
    def extract_terms(text: str) -> list[str]:
        """Extract conservative technical terms from text."""
        terms: list[str] = []

        for match in _TERM_RE.finditer(text):
            term = match.group(1).strip(".,;:!?()[]{}<>\"'")
            if ContextBuilder._is_useful_term(term):
                terms.append(term)

        words = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        for (left, _left_start, left_end), (right, right_start, _right_end) in zip(
            words, words[1:], strict=False
        ):
            between = text[left_end:right_start]
            if between.strip():
                continue
            if ContextBuilder._is_multiword_candidate(left, right):
                terms.append(f"{left} {right}")

        return ContextBuilder._dedupe(terms)

    @staticmethod
    def _is_multiword_candidate(left: str, right: str) -> bool:
        left = left.strip(".,;:!?()[]{}<>\"'")
        right = right.strip(".,;:!?()[]{}<>\"'")
        if not left or not right:
            return False
        if left in _STOPWORDS or right in _STOPWORDS:
            return False
        left_technical = ContextBuilder._is_useful_term(left)
        right_technical = ContextBuilder._is_useful_term(right)
        return left_technical and right_technical

    @staticmethod
    def _is_useful_term(term: str) -> bool:
        if len(term) < 2 or term in _STOPWORDS:
            return False
        if len(term) > 80:
            return False
        has_separator = any(char in term for char in "-_/.:@")
        has_digit = any(char.isdigit() for char in term)
        has_upper = any(char.isupper() for char in term)
        has_lower = any(char.islower() for char in term)
        is_acronym = bool(re.fullmatch(r"[A-Z]{2,}[a-z]*", term))
        is_mixed_case = has_upper and has_lower and not term.istitle()
        return has_separator or has_digit or is_acronym or is_mixed_case

    @staticmethod
    def _coerce_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        return [str(value)]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            item = str(item).strip()
            if not item:
                continue
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _limit_context(context: str, max_chars: int) -> str:
        if max_chars <= 0 or len(context) <= max_chars:
            return context
        marker = "\n[Context truncated to configured limit.]"
        return context[: max(0, max_chars - len(marker))].rstrip() + marker
