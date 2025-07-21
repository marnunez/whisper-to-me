"""
Single Instance Module

Ensures only one instance of Whisper-to-Me is running at a time.
"""

import atexit
import fcntl
import os
import sys
from pathlib import Path

from whisper_to_me.logger import get_logger


class SingleInstance:
    """
    Ensures only one instance of the application is running using file locking.

    Uses XDG_RUNTIME_DIR for automatic cleanup on logout/reboot.
    """

    def __init__(self):
        """Initialize single instance checker."""
        # Use XDG_RUNTIME_DIR if available, fallback to home directory
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if runtime_dir and Path(runtime_dir).exists():
            self.lockfile_path = Path(runtime_dir) / "whisper-to-me.lock"
        else:
            # Fallback for systems without XDG_RUNTIME_DIR
            self.lockfile_path = Path.home() / ".whisper-to-me.lock"

        self.lockfile = None
        self.logger = get_logger()

    def acquire(self) -> bool:
        """
        Try to acquire the single instance lock.

        Returns:
            True if lock acquired (first instance), False if already running
        """
        try:
            # Ensure parent directory exists
            self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)

            # Open or create lock file
            self.lockfile = open(self.lockfile_path, "w")

            # Try to acquire an exclusive lock (non-blocking)
            fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Register cleanup
            atexit.register(self.release)

            return True

        except OSError:
            # Lock is held by another instance
            if self.lockfile:
                self.lockfile.close()
                self.lockfile = None
            return False

    def release(self):
        """Release the single instance lock."""
        if self.lockfile:
            try:
                # Release the lock
                fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
                self.lockfile.close()
                # Try to remove the lock file (may fail if in runtime dir)
                try:
                    self.lockfile_path.unlink()
                except OSError:
                    pass
            except Exception:
                pass
            self.lockfile = None

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            self.logger.warning("Whisper-to-Me is already running!", "app")
            self.logger.info("Only one instance can run at a time.", "app")
            self.logger.info("\nTo stop the running instance:", "app")
            self.logger.info(
                "  - Check your system tray and right-click to quit", "app"
            )
            self.logger.info(
                "  - Or find the terminal running it and press Ctrl+C", "app"
            )
            self.logger.info(f"\nLock file location: {self.lockfile_path}", "app")
            sys.exit(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
