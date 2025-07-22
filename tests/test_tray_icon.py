"""Test tray icon functionality."""

import shutil
import tempfile
from unittest.mock import Mock, patch

from PIL import Image

from whisper_to_me import TrayIcon


class TestTrayIcon:
    """Test cases for tray icon functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_callbacks = {
            "on_quit": Mock(),
            "on_profile_change": Mock(),
            "get_profiles": Mock(return_value=["default", "work", "spanish"]),
            "get_current_profile": Mock(return_value="default"),
        }

        self.tray = TrayIcon(
            on_quit=self.mock_callbacks["on_quit"],
            on_profile_change=self.mock_callbacks["on_profile_change"],
            get_profiles=self.mock_callbacks["get_profiles"],
            get_current_profile=self.mock_callbacks["get_current_profile"],
        )

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test tray icon initialization."""
        assert self.tray.icon is None
        assert self.tray.is_recording is False
        assert self.tray._running is False
        assert self.tray.current_profile == "default"
        assert self.tray.on_quit_callback == self.mock_callbacks["on_quit"]

    def test_create_fallback_icon(self):
        """Test fallback icon creation."""
        # Test idle icon
        idle_icon = self.tray._create_fallback_icon(recording=False)
        assert isinstance(idle_icon, Image.Image)
        assert idle_icon.size == (32, 32)
        assert idle_icon.mode == "RGBA"

        # Test recording icon
        recording_icon = self.tray._create_fallback_icon(recording=True)
        assert isinstance(recording_icon, Image.Image)
        assert recording_icon.size == (32, 32)
        assert recording_icon.mode == "RGBA"

    @patch("pathlib.Path.exists")
    def test_create_image_fallback_when_no_icon(self, mock_exists):
        """Test image creation falls back when icon file doesn't exist."""
        mock_exists.return_value = False

        with patch.object(self.tray, "_create_fallback_icon") as mock_fallback:
            mock_fallback.return_value = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

            image = self.tray.create_image(recording=False)
            mock_fallback.assert_called_once_with(False)
            assert isinstance(image, Image.Image)

    def test_update_icon_status(self):
        """Test updating icon recording status."""
        # Mock the icon object
        self.tray.icon = Mock()

        # Test updating to recording
        self.tray.update_icon(recording=True)
        assert self.tray.is_recording is True

        # Test updating to idle
        self.tray.update_icon(recording=False)
        assert self.tray.is_recording is False

    def test_update_profile(self):
        """Test updating current profile."""
        # Mock the icon object
        self.tray.icon = Mock()

        with patch.object(self.tray, "create_menu") as mock_create_menu:
            mock_menu = Mock()
            mock_create_menu.return_value = mock_menu

            self.tray.update_profile("work")

            assert self.tray.current_profile == "work"
            assert self.tray.icon.title == "Whisper-to-Me (Profile: work)"
            mock_create_menu.assert_called_once()

    def test_profile_selection_callback(self):
        """Test profile selection triggers callback."""
        mock_icon = Mock()
        mock_item = Mock()

        self.tray.on_profile_select(mock_icon, mock_item, "spanish")

        self.mock_callbacks["on_profile_change"].assert_called_once_with("spanish")
        assert self.tray.current_profile == "spanish"

    def test_quit_callback(self):
        """Test quit triggers callback and stops icon."""
        mock_icon = Mock()
        mock_item = Mock()

        with patch.object(self.tray, "stop") as mock_stop:
            self.tray.on_quit(mock_icon, mock_item)

            mock_stop.assert_called_once()
            self.mock_callbacks["on_quit"].assert_called_once()

    def test_create_menu_structure(self):
        """Test menu structure creation."""
        menu = self.tray.create_menu()

        # Menu should be created
        assert menu is not None

        # Should call get_profiles and get_current_profile
        self.mock_callbacks["get_profiles"].assert_called()
        self.mock_callbacks["get_current_profile"].assert_called()

    def test_create_menu_with_multiple_profiles(self):
        """Test menu creation with multiple profiles."""
        # Mock multiple profiles
        self.mock_callbacks["get_profiles"].return_value = [
            "default",
            "work",
            "spanish",
            "quick",
        ]

        menu = self.tray.create_menu()
        assert menu is not None

    def test_create_menu_with_single_profile(self):
        """Test menu creation with only default profile."""
        # Mock single profile
        self.mock_callbacks["get_profiles"].return_value = ["default"]

        menu = self.tray.create_menu()
        assert menu is not None

    def test_refresh_menu(self):
        """Test manual menu refresh."""
        self.tray.icon = Mock()

        with patch.object(self.tray, "create_menu") as mock_create_menu:
            mock_menu = Mock()
            mock_create_menu.return_value = mock_menu

            self.tray.refresh_menu()

            mock_create_menu.assert_called_once()
            assert self.tray.icon.menu == mock_menu

    def test_profile_switch_handler_creation(self):
        """Test profile switch handler creation."""
        handler = self.tray._create_profile_switch_handler("test_profile")

        # Handler should be callable
        assert callable(handler)

        # Test handler execution
        mock_icon = Mock()
        mock_item = Mock()

        with patch.object(self.tray, "on_profile_select") as mock_select:
            handler(mock_icon, mock_item)
            mock_select.assert_called_once_with(mock_icon, mock_item, "test_profile")

    @patch("pystray.Icon")
    def test_run_creates_icon(self, mock_icon_class):
        """Test that run creates and starts the icon."""
        mock_icon_instance = Mock()
        mock_icon_class.return_value = mock_icon_instance

        with patch.object(self.tray, "create_menu") as mock_create_menu:
            with patch.object(self.tray, "create_image") as mock_create_image:
                mock_menu = Mock()
                mock_image = Mock()
                mock_create_menu.return_value = mock_menu
                mock_create_image.return_value = mock_image

                self.tray.run()

                # Should create icon with correct parameters
                mock_icon_class.assert_called_once_with(
                    "whisper-to-me",
                    mock_image,
                    "Whisper-to-Me (Profile: default)",
                    menu=mock_menu,
                )

                # Should run the icon
                mock_icon_instance.run.assert_called_once()
                assert self.tray.icon == mock_icon_instance

    @patch("threading.Thread")
    def test_start_creates_thread(self, mock_thread_class):
        """Test that start creates and starts a daemon thread."""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        self.tray.start()

        # Should create thread with correct target and daemon=True
        mock_thread_class.assert_called_once_with(target=self.tray.run, daemon=True)
        mock_thread.start.assert_called_once()

    def test_stop_sets_flags_and_stops_icon(self):
        """Test that stop sets running flag and stops icon."""
        # Set up icon
        self.tray.icon = Mock()
        self.tray._running = True

        self.tray.stop()

        assert self.tray._running is False
        self.tray.icon.stop.assert_called_once()

    def test_stop_handles_no_icon(self):
        """Test that stop handles case where icon is None."""
        self.tray.icon = None
        self.tray._running = True

        # Should not raise exception
        self.tray.stop()

        assert self.tray._running is False

    def test_callbacks_optional(self):
        """Test that tray works with None callbacks."""
        tray = TrayIcon()

        assert tray.on_quit_callback is None
        assert tray.on_profile_change_callback is None
        assert tray.get_profiles_callback is None
        assert tray.get_current_profile_callback is None

        # Should handle None callbacks gracefully
        mock_icon = Mock()
        mock_item = Mock()

        # These should not raise exceptions
        tray.on_quit(mock_icon, mock_item)
        tray.on_profile_select(mock_icon, mock_item, "test")

    def test_icon_creation_with_real_assets(self):
        """Test icon creation with actual asset files."""
        # The real icons should exist in the project
        image = self.tray.create_image(recording=False)
        assert isinstance(image, Image.Image)
        assert image.size == (32, 32)
        assert image.mode == "RGBA"

        # Test recording version
        recording_image = self.tray.create_image(recording=True)
        assert isinstance(recording_image, Image.Image)
        assert recording_image.size == (32, 32)
        assert recording_image.mode == "RGBA"

        # Images should be different when recording status changes
        assert image.tobytes() != recording_image.tobytes()

    def test_error_handling_in_update_methods(self):
        """Test error handling in update methods."""
        # Mock icon that raises exceptions
        self.tray.icon = Mock()
        self.tray.icon.icon = Mock(side_effect=Exception("Test error"))

        # Should not raise exception, just print error
        self.tray.update_icon(recording=True)

        # Mock menu assignment that raises exception
        self.tray.icon.menu = Mock(side_effect=Exception("Menu error"))

        # Should not raise exception
        with patch.object(self.tray, "create_menu", return_value=Mock()):
            self.tray.update_profile("test")
            self.tray.refresh_menu()


class TestTrayIconIntegration:
    """Integration tests for tray icon with real-like scenarios."""

    def test_full_profile_switching_workflow(self):
        """Test complete profile switching workflow."""
        profiles = ["default", "work", "spanish"]
        current_profile = "default"

        def mock_get_profiles():
            return profiles

        def mock_get_current_profile():
            return current_profile

        def mock_profile_change(profile_name):
            nonlocal current_profile
            current_profile = profile_name

        tray = TrayIcon(
            get_profiles=mock_get_profiles,
            get_current_profile=mock_get_current_profile,
            on_profile_change=mock_profile_change,
        )

        # Initial state
        assert tray.current_profile == "default"

        # Switch to work profile
        tray.on_profile_select(Mock(), Mock(), "work")
        assert current_profile == "work"
        assert tray.current_profile == "work"

        # Switch to spanish profile
        tray.on_profile_select(Mock(), Mock(), "spanish")
        assert current_profile == "spanish"
        assert tray.current_profile == "spanish"

    def test_recording_status_visual_feedback(self):
        """Test that recording status changes visual representation."""
        tray = TrayIcon()

        # Create icons for different states
        idle_icon = tray.create_image(recording=False)
        recording_icon = tray.create_image(recording=True)

        # Icons should be different
        assert idle_icon.tobytes() != recording_icon.tobytes()

        # Both should be valid images
        assert isinstance(idle_icon, Image.Image)
        assert isinstance(recording_icon, Image.Image)
        assert idle_icon.size == recording_icon.size == (32, 32)
