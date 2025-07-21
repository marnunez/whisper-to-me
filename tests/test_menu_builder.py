"""Test menu builder functionality."""

from unittest.mock import Mock, patch

from whisper_to_me import (
    DeviceMenuFormatter,
    MenuBuilder,
    ProfileMenuFormatter,
    TrayMenuBuilder,
)


class TestMenuBuilder:
    """Test MenuBuilder functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.builder = MenuBuilder()

    def test_init(self):
        """Test MenuBuilder initialization."""
        assert self.builder.menu_items == []

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_header(self, mock_pystray):
        """Test add_header method."""
        self.builder.add_header("Test Header", enabled=True)

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once_with("Test Header", None, enabled=True)

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_header_default_disabled(self, mock_pystray):
        """Test add_header with default enabled=False."""
        self.builder.add_header("Test Header")

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once_with(
            "Test Header", None, enabled=False
        )

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_default_header(self, mock_pystray):
        """Test add_default_header method."""
        self.builder.add_default_header("Default Header")

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once_with(
            "Default Header", None, default=True, enabled=False
        )

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_info_item(self, mock_pystray):
        """Test add_info_item method."""
        self.builder.add_info_item("Info text")

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once_with("Info text", None, enabled=False)

    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_add_separator(self, mock_menu):
        """Test add_separator method."""
        mock_menu.SEPARATOR = "SEPARATOR"

        self.builder.add_separator()

        assert len(self.builder.menu_items) == 1
        assert self.builder.menu_items[0] == "SEPARATOR"

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_action_item(self, mock_pystray):
        """Test add_action_item method."""
        mock_handler = Mock()

        self.builder.add_action_item("Action", mock_handler)

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once_with("Action", mock_handler)

    @patch("whisper_to_me.menu_builder.pystray")
    def test_add_submenu(self, mock_pystray):
        """Test add_submenu method."""
        submenu_items = [Mock(), Mock()]

        self.builder.add_submenu("Submenu", submenu_items)

        assert len(self.builder.menu_items) == 1
        mock_pystray.MenuItem.assert_called_once()
        # Check that Menu was created with submenu items
        mock_pystray.Menu.assert_called_once_with(*submenu_items)

    @patch("whisper_to_me.menu_builder.pystray")
    def test_build(self, mock_pystray):
        """Test build method."""
        # Add some items
        mock_pystray.Menu.SEPARATOR = "SEPARATOR"
        self.builder.add_header("Header")
        self.builder.add_separator()
        self.builder.add_action_item("Action", Mock())

        self.builder.build()

        # Should create Menu with all items
        assert mock_pystray.Menu.called
        assert len(self.builder.menu_items) == 3

    @patch("whisper_to_me.menu_builder.pystray")
    def test_clear(self, mock_pystray):
        """Test clear method."""
        # Add some items
        self.builder.add_header("Header")
        self.builder.add_action_item("Action", Mock())

        assert len(self.builder.menu_items) == 2

        # Clear items
        self.builder.clear()

        assert len(self.builder.menu_items) == 0


class TestProfileMenuFormatter:
    """Test ProfileMenuFormatter functionality."""

    def test_init(self):
        """Test ProfileMenuFormatter initialization."""
        get_profiles = Mock(return_value=["default", "work"])
        get_current = Mock(return_value="default")
        handler = Mock()

        formatter = ProfileMenuFormatter(get_profiles, get_current, handler)

        assert formatter.get_profiles == get_profiles
        assert formatter.get_current_profile == get_current
        assert formatter.profile_switch_handler == handler

    @patch("whisper_to_me.menu_builder.pystray")
    def test_create_profile_menu_items_single_profile(self, mock_pystray):
        """Test menu creation with single profile."""
        get_profiles = Mock(return_value=["default"])
        get_current = Mock(return_value="default")
        handler = Mock()

        formatter = ProfileMenuFormatter(get_profiles, get_current, handler)
        result = formatter.create_profile_menu_items()

        # Should return empty list for single profile
        assert result == []

    @patch("whisper_to_me.menu_builder.pystray")
    def test_create_profile_menu_items_multiple_profiles(self, mock_pystray):
        """Test menu creation with multiple profiles."""
        profiles = ["default", "work", "gaming"]
        get_profiles = Mock(return_value=profiles)
        get_current = Mock(return_value="work")

        # Create handler that returns different functions for each profile
        handlers = {profile: Mock(name=f"handler_{profile}") for profile in profiles}

        def handler(p):
            return handlers[p]

        formatter = ProfileMenuFormatter(get_profiles, get_current, handler)
        result = formatter.create_profile_menu_items()

        # Should create menu items for each profile
        assert len(result) == 3
        assert mock_pystray.MenuItem.call_count == 3

        # Check the calls
        calls = mock_pystray.MenuItem.call_args_list
        assert calls[0][0] == ("○ default", handlers["default"])
        assert calls[1][0] == ("● work", handlers["work"])  # Current profile
        assert calls[2][0] == ("○ gaming", handlers["gaming"])


class TestDeviceMenuFormatter:
    """Test DeviceMenuFormatter functionality."""

    def test_init(self):
        """Test DeviceMenuFormatter initialization."""
        get_devices = Mock(return_value=[])
        get_current = Mock(return_value=None)
        handler = Mock()

        formatter = DeviceMenuFormatter(get_devices, get_current, handler)

        assert formatter.get_devices == get_devices
        assert formatter.get_current_device == get_current
        assert formatter.device_switch_handler == handler

    @patch("whisper_to_me.menu_builder.pystray")
    def test_create_device_menu_items_no_devices(self, mock_pystray):
        """Test menu creation with no devices."""
        get_devices = Mock(return_value=[])
        get_current = Mock(return_value=None)
        handler = Mock()

        formatter = DeviceMenuFormatter(get_devices, get_current, handler)
        result = formatter.create_device_menu_items()

        # Should return empty list for no devices
        assert result == []

    @patch("whisper_to_me.menu_builder.pystray")
    def test_create_device_menu_items_single_device(self, mock_pystray):
        """Test menu creation with single device."""
        device = {"id": 1, "name": "USB Mic", "hostapi_name": "ALSA"}
        get_devices = Mock(return_value=[device])
        get_current = Mock(return_value=device)
        handler = Mock()

        formatter = DeviceMenuFormatter(get_devices, get_current, handler)
        result = formatter.create_device_menu_items()

        # Should return empty list for single device
        assert result == []


class TestTrayMenuBuilder:
    """Test TrayMenuBuilder functionality."""

    def test_init(self):
        """Test TrayMenuBuilder initialization."""
        builder = TrayMenuBuilder()

        assert isinstance(builder.menu_builder, MenuBuilder)
        assert builder.profile_formatter is None
        assert builder.device_formatter is None

    def test_set_profile_formatter(self):
        """Test setting profile formatter."""
        builder = TrayMenuBuilder()
        formatter = Mock(spec=ProfileMenuFormatter)

        builder.set_profile_formatter(formatter)

        assert builder.profile_formatter == formatter

    def test_set_device_formatter(self):
        """Test setting device formatter."""
        builder = TrayMenuBuilder()
        formatter = Mock(spec=DeviceMenuFormatter)

        builder.set_device_formatter(formatter)

        assert builder.device_formatter == formatter

    @patch("whisper_to_me.menu_builder.pystray")
    def test_build_complete_menu_basic(self, mock_pystray):
        """Test building complete menu with basic info."""
        mock_pystray.Menu.SEPARATOR = "SEPARATOR"

        builder = TrayMenuBuilder()
        on_quit = Mock()

        builder.build_complete_menu(
            current_profile="default", current_device=None, on_quit=on_quit
        )

        # Should have created menu with header and quit
        assert mock_pystray.Menu.called
        assert mock_pystray.MenuItem.call_count >= 3  # Header, profile info, quit

    @patch("whisper_to_me.menu_builder.pystray")
    def test_build_complete_menu_with_device(self, mock_pystray):
        """Test building menu with device info."""
        mock_pystray.Menu.SEPARATOR = "SEPARATOR"

        builder = TrayMenuBuilder()
        device = {"name": "USB Microphone", "id": 1}

        builder.build_complete_menu(
            current_profile="default", current_device=device, on_quit=Mock()
        )

        # Should include device info
        info_calls = [
            call
            for call in mock_pystray.MenuItem.call_args_list
            if "Device: USB Microphone" in str(call)
        ]
        assert len(info_calls) > 0

    @patch("whisper_to_me.menu_builder.pystray")
    def test_build_complete_menu_with_formatters(self, mock_pystray):
        """Test building menu with profile and device formatters."""
        mock_pystray.Menu.SEPARATOR = "SEPARATOR"

        builder = TrayMenuBuilder()

        # Set up profile formatter
        profile_formatter = Mock(spec=ProfileMenuFormatter)
        profile_formatter.create_profile_menu_items.return_value = [Mock(), Mock()]
        builder.set_profile_formatter(profile_formatter)

        # Set up device formatter
        device_formatter = Mock(spec=DeviceMenuFormatter)
        device_formatter.create_device_menu_items.return_value = [Mock(), Mock()]
        builder.set_device_formatter(device_formatter)

        builder.build_complete_menu(
            current_profile="work",
            current_device={"name": "Test Device"},
            on_quit=Mock(),
        )

        # Should have called formatters
        profile_formatter.create_profile_menu_items.assert_called_once()
        device_formatter.create_device_menu_items.assert_called_once()

        # Should have created submenus
        submenu_calls = [
            call
            for call in mock_pystray.MenuItem.call_args_list
            if "Switch Profile" in str(call) or "Select Audio Device" in str(call)
        ]
        assert len(submenu_calls) == 2
