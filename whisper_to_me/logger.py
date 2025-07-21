"""
Logger Utility Module

Provides centralized logging functionality to replace scattered print statements
throughout the codebase with structured, configurable logging.
"""

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TextIO


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


class Logger:
    """
    Centralized logger with configurable output and formatting.

    Features:
    - Multiple log levels with filtering
    - Configurable output streams (stdout, stderr, file)
    - Timestamp formatting
    - Category-based logging
    - Emoji icons for different message types
    - Thread-safe output
    """

    # Emoji icons for different log types
    ICONS = {
        "success": "✓",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "debug": "🐛",
        "recording": "🎤",
        "processing": "🔄",
        "device": "📱",
        "profile": "👤",
        "config": "⚙️",
        "language": "🌐",
        "model": "🧠",
        "key": "⌨️",
        "tray": "📟",
        "shutdown": "🛑",
        "startup": "🚀",
        "transcription": "📝",
    }

    def __init__(
        self,
        min_level: LogLevel = LogLevel.INFO,
        output_stream: TextIO | None = None,
        log_file: Path | None = None,
        include_timestamps: bool = False,
        include_categories: bool = True,
    ):
        """
        Initialize the logger.

        Args:
            min_level: Minimum log level to output
            output_stream: Output stream (default: stdout)
            log_file: Optional file to write logs to
            include_timestamps: Whether to include timestamps
            include_categories: Whether to include category labels
        """
        self.min_level = min_level
        self.output_stream = output_stream or sys.stdout
        self.log_file = log_file
        self.include_timestamps = include_timestamps
        self.include_categories = include_categories

        # Create log file if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _format_message(
        self,
        level: LogLevel,
        message: str,
        category: str | None = None,
        icon: str | None = None,
    ) -> str:
        """Format a log message with optional timestamp and category."""
        parts = []

        # Add timestamp if enabled
        if self.include_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")

        # Add icon if provided
        if icon:
            parts.append(icon)

        # Add category if enabled and provided
        if self.include_categories and category:
            parts.append(f"[{category.upper()}]")

        # Add the message
        parts.append(message)

        return " ".join(parts)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message should be logged based on the minimum level."""
        return level.value >= self.min_level.value

    def _write_message(self, formatted_message: str) -> None:
        """Write message to output stream and optional log file."""
        # Write to output stream
        print(formatted_message, file=self.output_stream)

        # Write to log file if specified
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {formatted_message}\n")
            except Exception:
                # Don't let log file errors break the application
                pass

    def log(
        self,
        level: LogLevel,
        message: str,
        category: str | None = None,
        icon: str | None = None,
    ) -> None:
        """
        Log a message at the specified level.

        Args:
            level: Log level
            message: Message to log
            category: Optional category for organization
            icon: Optional icon key or emoji
        """
        if not self._should_log(level):
            return

        # Resolve icon
        if icon and icon in self.ICONS:
            icon = self.ICONS[icon]

        formatted_message = self._format_message(level, message, category, icon)
        self._write_message(formatted_message)

    def debug(
        self, message: str, category: str | None = None, icon: str = "debug"
    ) -> None:
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message, category, icon)

    def info(
        self, message: str, category: str | None = None, icon: str = "info"
    ) -> None:
        """Log an info message."""
        self.log(LogLevel.INFO, message, category, icon)

    def warning(
        self, message: str, category: str | None = None, icon: str = "warning"
    ) -> None:
        """Log a warning message."""
        self.log(LogLevel.WARNING, message, category, icon)

    def error(
        self, message: str, category: str | None = None, icon: str = "error"
    ) -> None:
        """Log an error message."""
        self.log(LogLevel.ERROR, message, category, icon)

    def critical(
        self, message: str, category: str | None = None, icon: str = "error"
    ) -> None:
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, message, category, icon)

    # Specialized logging methods for common use cases
    def success(self, message: str, category: str | None = None) -> None:
        """Log a success message."""
        self.log(LogLevel.INFO, message, category, "success")

    def recording_started(self) -> None:
        """Log recording start."""
        self.log(LogLevel.INFO, "Recording started...", "audio", "recording")

    def recording_stopped(self, duration: float, samples: int) -> None:
        """Log recording stop."""
        message = f"Recording stopped. Captured {samples} samples ({duration:.2f}s)"
        self.log(LogLevel.INFO, message, "audio", "processing")

    def transcription_completed(
        self,
        text: str,
        language: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Log transcription completion."""
        message = "Transcription completed"
        if language:
            message += f" (language: {language}"
            if confidence:
                message += f", confidence: {confidence:.2f}"
            message += ")"
        self.log(LogLevel.INFO, message, "speech", "transcription")

        if text.strip():
            self.debug(f"Transcribed text: '{text}'", "speech")

    def device_switched(self, device_name: str) -> None:
        """Log device switch."""
        self.log(
            LogLevel.INFO,
            f"Switched to audio device: {device_name}",
            "device",
            "device",
        )

    def profile_switched(self, profile_name: str) -> None:
        """Log profile switch."""
        self.log(
            LogLevel.INFO, f"Switched to profile: {profile_name}", "profile", "profile"
        )

    def model_loaded(self, model_name: str, device: str) -> None:
        """Log model loading."""
        message = f"Loading {model_name} model on {device}..."
        self.log(LogLevel.INFO, message, "speech", "model")

    def application_startup(self, profile: str) -> None:
        """Log application startup."""
        self.log(
            LogLevel.INFO,
            f"Whisper-to-Me starting (profile: {profile})",
            "app",
            "startup",
        )

    def application_shutdown(self) -> None:
        """Log application shutdown."""
        self.log(LogLevel.INFO, "Shutting down...", "app", "shutdown")
        self.log(LogLevel.INFO, "Goodbye!", "app")

    def hotkey_info(
        self, trigger_key: str, mode: str, discard_key: str | None = None
    ) -> None:
        """Log hotkey configuration."""
        if mode == "tap-mode":
            message = f"Ready! Tap {trigger_key} to start/stop recording"
            if discard_key:
                message += f". Press {discard_key} while recording to discard"
        else:
            message = f"Ready! Press and hold {trigger_key} to record"

        self.log(LogLevel.INFO, message, "hotkey", "key")


# Global logger instance
_global_logger: Logger | None = None


def get_logger() -> Logger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def setup_logger(
    min_level: LogLevel = LogLevel.INFO,
    output_stream: TextIO | None = None,
    log_file: Path | None = None,
    include_timestamps: bool = False,
    include_categories: bool = True,
) -> Logger:
    """
    Setup the global logger with custom configuration.

    Args:
        min_level: Minimum log level to output
        output_stream: Output stream (default: stdout)
        log_file: Optional file to write logs to
        include_timestamps: Whether to include timestamps
        include_categories: Whether to include category labels

    Returns:
        Configured logger instance
    """
    global _global_logger
    _global_logger = Logger(
        min_level=min_level,
        output_stream=output_stream,
        log_file=log_file,
        include_timestamps=include_timestamps,
        include_categories=include_categories,
    )
    return _global_logger
