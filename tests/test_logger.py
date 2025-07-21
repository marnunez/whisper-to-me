"""Test logger functionality."""

import io
import tempfile
from pathlib import Path

from whisper_to_me.logger import Logger, LogLevel, get_logger, setup_logger


class TestLogger:
    """Test cases for logger functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.output_stream = io.StringIO()
        self.logger = Logger(
            min_level=LogLevel.DEBUG,
            output_stream=self.output_stream,
            include_timestamps=False,
            include_categories=True,
        )

    def test_log_levels(self):
        """Test different log levels."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")

        output = self.output_stream.getvalue()
        assert "🐛 Debug message" in output  # No level in default format
        assert "ℹ️ Info message" in output
        assert "⚠️ Warning message" in output
        assert "❌ Error message" in output
        assert "❌ Critical message" in output

    def test_log_level_filtering(self):
        """Test that log level filtering works."""
        logger = Logger(
            min_level=LogLevel.WARNING,
            output_stream=self.output_stream,
            include_timestamps=False,
        )

        logger.debug("Should not appear")
        logger.info("Should not appear")
        logger.warning("Should appear")
        logger.error("Should appear")

        output = self.output_stream.getvalue()
        assert "Should not appear" not in output
        assert "Should appear" in output

    def test_custom_icons(self):
        """Test custom icon usage."""
        self.logger.log(LogLevel.INFO, "Test message", icon="success")
        self.logger.log(LogLevel.INFO, "Test message", icon="device")
        self.logger.log(LogLevel.INFO, "Test message", icon="🚀")  # Direct emoji

        output = self.output_stream.getvalue()
        assert "✓" in output  # success icon
        assert "📱" in output  # device icon
        assert "🚀" in output  # direct emoji

    def test_categories(self):
        """Test category inclusion."""
        self.logger.info("Test message", category="audio")
        self.logger.error("Error message", category="config")

        output = self.output_stream.getvalue()
        assert "[AUDIO]" in output
        assert "[CONFIG]" in output

    def test_no_categories(self):
        """Test logger without categories."""
        logger = Logger(output_stream=self.output_stream, include_categories=False)

        logger.info("Test message", category="audio")
        output = self.output_stream.getvalue()
        assert "[AUDIO]" not in output
        assert "Test message" in output

    def test_specialized_logging_methods(self):
        """Test specialized logging methods."""
        self.logger.success("Operation completed")
        self.logger.recording_started()
        self.logger.recording_stopped(2.5, 1024)
        self.logger.transcription_completed("Hello world", "en", 0.95)
        self.logger.device_switched("Test Device")
        self.logger.profile_switched("work")

        output = self.output_stream.getvalue()
        assert "✓" in output  # success icon
        assert "🎤" in output  # recording icon
        assert "🔄" in output  # processing icon
        assert "📝" in output  # transcription icon
        assert "📱" in output  # device icon
        assert "👤" in output  # profile icon
        assert "Operation completed" in output
        assert "Recording started" in output
        assert "2.50s" in output  # Logger formats to 2 decimal places
        assert "Hello world" in output
        assert "Test Device" in output
        assert "work" in output

    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            logger = Logger(
                output_stream=self.output_stream,
                log_file=log_file,
                include_timestamps=False,
            )

            logger.info("Test message")

            # Check console output
            assert "Test message" in self.output_stream.getvalue()

            # Check file output
            assert log_file.exists()
            with open(log_file, encoding="utf-8") as f:
                file_content = f.read()
            assert "Test message" in file_content

    def test_timestamps(self):
        """Test timestamp inclusion."""
        logger = Logger(
            output_stream=self.output_stream,
            include_timestamps=True,
            include_categories=False,
        )

        logger.info("Test message")
        output = self.output_stream.getvalue()

        # Should contain timestamp pattern [HH:MM:SS]
        import re

        assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", output)

    def test_global_logger_functions(self):
        """Test global logger functions."""
        # Test getting default global logger
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2  # Should be the same instance

        # Test setting up global logger
        output_stream = io.StringIO()
        setup_logger(min_level=LogLevel.ERROR, output_stream=output_stream)

        global_logger = get_logger()
        global_logger.info("Should not appear")  # Below ERROR level
        global_logger.error("Should appear")

        output = output_stream.getvalue()
        assert "Should not appear" not in output
        assert "Should appear" in output

    def test_model_and_startup_logging(self):
        """Test specialized startup and model logging."""
        self.logger.model_loaded("large-v3", "cuda")
        self.logger.application_startup("default")
        self.logger.hotkey_info("<scroll_lock>", "push-to-talk")
        self.logger.hotkey_info("<caps_lock>", "tap-mode", "<esc>")
        self.logger.application_shutdown()

        output = self.output_stream.getvalue()
        assert "🧠" in output  # model icon
        assert "🚀" in output  # startup icon
        assert "⌨️" in output  # key icon
        assert "🛑" in output  # shutdown icon
        assert "large-v3 model on cuda" in output
        assert "press and hold <scroll_lock>" in output.lower()
        assert "tap <caps_lock>" in output.lower()
        assert "<esc>" in output
        assert "Goodbye!" in output
