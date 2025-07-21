"""Test menu builder functionality."""

from unittest.mock import Mock, patch

from whisper_to_me.menu_builder import (
    MenuBuilder,
    ProfileMenuFormatter,
    DeviceMenuFormatter,
    TrayMenuBuilder,
)


class TestMenuBuilder:
    """Test MenuBuilder functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.builder = MenuBuilder()

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_init(self, mock_menu_item):
        """Test MenuBuilder initialization."""
        builder = MenuBuilder()
        assert builder.menu_items == []

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_add_header(self, mock_menu_item):
        """Test add_header method."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item

        self.builder.add_header("Test Header", enabled=True)

        mock_menu_item.assert_called_once_with("Test Header", None, enabled=True)
        assert mock_item in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_add_header_default_disabled(self, mock_menu_item):
        """Test add_header with default enabled=False."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item

        self.builder.add_header("Test Header")

        mock_menu_item.assert_called_once_with("Test Header", None, enabled=False)

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_add_default_header(self, mock_menu_item):
        """Test add_default_header method."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item

        self.builder.add_default_header("Default Header")

        mock_menu_item.assert_called_once_with(
            "Default Header", None, default=True, enabled=False
        )
        assert mock_item in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_add_info_item(self, mock_menu_item):
        """Test add_info_item method."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item

        self.builder.add_info_item("Info text")

        mock_menu_item.assert_called_once_with("Info text", None, enabled=False)
        assert mock_item in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_add_separator(self, mock_menu):
        """Test add_separator method."""
        mock_separator = Mock()
        mock_menu.SEPARATOR = mock_separator

        self.builder.add_separator()

        assert mock_separator in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_add_action_item(self, mock_menu_item):
        """Test add_action_item method."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item
        mock_handler = Mock()

        self.builder.add_action_item("Action", mock_handler)

        mock_menu_item.assert_called_once_with("Action", mock_handler)
        assert mock_item in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_add_submenu(self, mock_menu, mock_menu_item):
        """Test add_submenu method."""
        mock_item = Mock()
        mock_menu_item.return_value = mock_item
        mock_submenu = Mock()
        mock_menu.return_value = mock_submenu

        submenu_items = [Mock(), Mock()]
        self.builder.add_submenu("Submenu", submenu_items)

        mock_menu.assert_called_once_with(*submenu_items)
        mock_menu_item.assert_called_once_with("Submenu", mock_submenu)
        assert mock_item in self.builder.menu_items

    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_build(self, mock_menu):
        """Test build method."""
        mock_built_menu = Mock()
        mock_menu.return_value = mock_built_menu

        # Add some items
        item1 = Mock()
        item2 = Mock()
        self.builder.menu_items = [item1, item2]

        result = self.builder.build()

        mock_menu.assert_called_once_with(item1, item2)
        assert result == mock_built_menu

    def test_clear(self):
        """Test clear method."""
        # Add some items
        self.builder.menu_items = [Mock(), Mock(), Mock()]

        self.builder.clear()

        assert self.builder.menu_items == []

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_complex_menu_building(self, mock_menu, mock_menu_item):
        """Test building a complex menu with various items."""
        # Mock return values
        mock_menu_item.side_effect = lambda *args, **kwargs: Mock()
        mock_menu.SEPARATOR = Mock()
        mock_menu.return_value = Mock()

        handler = Mock()
        submenu_items = [Mock()]

        # Build a complex menu
        self.builder.add_default_header("App")
        self.builder.add_info_item("Status: Running")
        self.builder.add_separator()
        self.builder.add_submenu("Options", submenu_items)
        self.builder.add_separator()
        self.builder.add_action_item("Quit", handler)

        result = self.builder.build()

        # Should have all items
        assert len(self.builder.menu_items) == 6
        assert result is not None


class TestProfileMenuFormatter:
    """Test ProfileMenuFormatter functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.get_profiles = Mock()
        self.get_current_profile = Mock()
        self.profile_switch_handler = Mock()

        self.formatter = ProfileMenuFormatter(
            self.get_profiles, self.get_current_profile, self.profile_switch_handler
        )

    def test_init(self):
        """Test ProfileMenuFormatter initialization."""
        assert self.formatter.get_profiles == self.get_profiles
        assert self.formatter.get_current_profile == self.get_current_profile
        assert self.formatter.profile_switch_handler == self.profile_switch_handler

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_profile_menu_items_single_profile(self, mock_menu_item):
        """Test create_profile_menu_items with single profile."""
        self.get_profiles.return_value = ["default"]

        result = self.formatter.create_profile_menu_items()

        assert result == []
        mock_menu_item.assert_not_called()

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_profile_menu_items_no_profiles(self, mock_menu_item):
        """Test create_profile_menu_items with no profiles."""
        self.get_profiles.return_value = []

        result = self.formatter.create_profile_menu_items()

        assert result == []
        mock_menu_item.assert_not_called()

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_profile_menu_items_multiple_profiles(self, mock_menu_item):
        """Test create_profile_menu_items with multiple profiles."""
        profiles = ["default", "work", "gaming"]
        self.get_profiles.return_value = profiles
        self.get_current_profile.return_value = "work"

        # Mock menu items
        mock_items = [Mock(), Mock(), Mock()]
        mock_menu_item.side_effect = mock_items

        # Mock profile switch handlers
        mock_handlers = [Mock(), Mock(), Mock()]
        self.profile_switch_handler.side_effect = mock_handlers

        result = self.formatter.create_profile_menu_items()

        assert len(result) == 3
        assert result == mock_items

        # Check calls to profile_switch_handler
        assert self.profile_switch_handler.call_count == 3
        self.profile_switch_handler.assert_any_call("default")
        self.profile_switch_handler.assert_any_call("work")
        self.profile_switch_handler.assert_any_call("gaming")

        # Check menu item creation with correct markers
        actual_calls = mock_menu_item.call_args_list
        assert len(actual_calls) == 3

        # Check display names have correct markers
        for i, call in enumerate(actual_calls):
            args, kwargs = call
            display_name = args[0]
            if "work" in display_name:
                assert "●" in display_name  # Current profile
            else:
                assert "○" in display_name  # Other profiles

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_profile_menu_items_order_preservation(self, mock_menu_item):
        """Test that profile order is preserved."""
        profiles = ["zebra", "alpha", "beta"]
        self.get_profiles.return_value = profiles
        self.get_current_profile.return_value = "alpha"

        mock_menu_item.side_effect = [Mock(), Mock(), Mock()]
        self.profile_switch_handler.side_effect = [Mock(), Mock(), Mock()]

        self.formatter.create_profile_menu_items()

        # Should maintain original order, not alphabetical
        calls = self.profile_switch_handler.call_args_list
        assert [call[0][0] for call in calls] == ["zebra", "alpha", "beta"]


class TestDeviceMenuFormatter:
    """Test DeviceMenuFormatter functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.get_devices = Mock()
        self.get_current_device = Mock()
        self.device_switch_handler = Mock()

        self.formatter = DeviceMenuFormatter(
            self.get_devices, self.get_current_device, self.device_switch_handler
        )

    def test_init(self):
        """Test DeviceMenuFormatter initialization."""
        assert self.formatter.get_devices == self.get_devices
        assert self.formatter.get_current_device == self.get_current_device
        assert self.formatter.device_switch_handler == self.device_switch_handler

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_menu_items_no_devices(self, mock_menu_item):
        """Test create_device_menu_items with no devices."""
        self.get_devices.return_value = []

        result = self.formatter.create_device_menu_items()

        assert result == []
        mock_menu_item.assert_not_called()

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_menu_items_single_device(self, mock_menu_item):
        """Test create_device_menu_items with single device."""
        devices = [{"id": 1, "name": "Device 1", "hostapi_name": "ALSA"}]
        self.get_devices.return_value = devices

        result = self.formatter.create_device_menu_items()

        assert result == []
        mock_menu_item.assert_not_called()

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_menu_items_single_hostapi(self, mock_menu_item):
        """Test create_device_menu_items with single host API."""
        devices = [
            {"id": 1, "name": "Device 1", "hostapi_name": "ALSA"},
            {"id": 2, "name": "Device 2", "hostapi_name": "ALSA"},
        ]
        self.get_devices.return_value = devices
        self.get_current_device.return_value = {"id": 1, "name": "Device 1"}

        mock_items = [Mock(), Mock()]
        mock_menu_item.side_effect = mock_items

        mock_handlers = [Mock(), Mock()]
        self.device_switch_handler.side_effect = mock_handlers

        result = self.formatter.create_device_menu_items()

        assert len(result) == 2
        assert result == mock_items

        # Should not create submenus for single host API
        mock_menu_item.assert_any_call("✓ Device 1", mock_handlers[0])
        mock_menu_item.assert_any_call("Device 2", mock_handlers[1])

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    @patch("whisper_to_me.menu_builder.pystray.Menu")
    def test_create_device_menu_items_multiple_hostapis(
        self, mock_menu, mock_menu_item
    ):
        """Test create_device_menu_items with multiple host APIs."""
        devices = [
            {"id": 1, "name": "ALSA Device 1", "hostapi_name": "ALSA"},
            {"id": 2, "name": "ALSA Device 2", "hostapi_name": "ALSA"},
            {
                "id": 3,
                "name": "JACK Device",
                "hostapi_name": "JACK Audio Connection Kit",
            },
        ]
        self.get_devices.return_value = devices
        self.get_current_device.return_value = {"id": 2, "name": "ALSA Device 2"}

        # Mock device items (need more for submenus)
        mock_device_items = [
            Mock() for _ in range(10)
        ]  # More items to avoid StopIteration
        mock_menu_item.side_effect = mock_device_items

        # Mock submenus
        mock_alsa_submenu = Mock()
        mock_jack_submenu = Mock()
        mock_menu.side_effect = [mock_alsa_submenu, mock_jack_submenu]

        # Mock handlers
        self.device_switch_handler.side_effect = [Mock(), Mock(), Mock()]

        result = self.formatter.create_device_menu_items()

        # Should create submenu items for each host API
        assert len(result) == 2  # Two host API submenus

        # Should have called mock_menu to create submenus
        assert mock_menu.call_count == 2

    def test_group_devices_by_hostapi(self):
        """Test _group_devices_by_hostapi method."""
        devices = [
            {"id": 1, "name": "Device 1", "hostapi_name": "ALSA"},
            {"id": 2, "name": "Device 2", "hostapi_name": "JACK Audio Connection Kit"},
            {"id": 3, "name": "Device 3", "hostapi_name": "ALSA"},
            {"id": 4, "name": "Device 4"},  # No hostapi_name
        ]

        result = self.formatter._group_devices_by_hostapi(devices)

        expected = {
            "ALSA": [devices[0], devices[2]],
            "JACK Audio Connection Kit": [devices[1]],
            "Unknown": [devices[3]],
        }

        assert result == expected

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_items_basic(self, mock_menu_item):
        """Test _create_device_items method."""
        devices = [{"id": 1, "name": "Device 1"}, {"id": 2, "name": "Device 2"}]
        current_device = {"id": 1, "name": "Device 1"}

        mock_items = [Mock(), Mock()]
        mock_menu_item.side_effect = mock_items

        mock_handlers = [Mock(), Mock()]
        self.device_switch_handler.side_effect = mock_handlers

        result = self.formatter._create_device_items(devices, current_device, False)

        assert len(result) == 2
        assert result == mock_items

        # Check correct display names and handlers
        mock_menu_item.assert_any_call(
            "✓ Device 1", mock_handlers[0]
        )  # Current device marked
        mock_menu_item.assert_any_call("Device 2", mock_handlers[1])

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_items_long_names(self, mock_menu_item):
        """Test _create_device_items with long device names."""
        devices = [
            {
                "id": 1,
                "name": "Very Long Device Name That Should Be Truncated Because It Is Too Long",
            },
            {"id": 2, "name": "Device 2"},
        ]
        current_device = None

        mock_items = [Mock(), Mock()]
        mock_menu_item.side_effect = mock_items
        self.device_switch_handler.side_effect = [Mock(), Mock()]

        # Test non-nested (max 40 chars)
        self.formatter._create_device_items(devices, current_device, False)

        calls = mock_menu_item.call_args_list
        truncated_name = calls[0][0][0]  # First argument of first call
        assert len(truncated_name) <= 40
        assert "..." in truncated_name

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_items_nested_shorter_limit(self, mock_menu_item):
        """Test _create_device_items with is_nested=True (shorter limit)."""
        devices = [
            {"id": 1, "name": "Medium Length Device Name That Should Be Truncated"}
        ]
        current_device = None

        mock_menu_item.return_value = Mock()
        self.device_switch_handler.return_value = Mock()

        # Test nested (max 35 chars)
        self.formatter._create_device_items(devices, current_device, True)

        calls = mock_menu_item.call_args_list
        truncated_name = calls[0][0][0]
        assert len(truncated_name) <= 35
        assert "..." in truncated_name

    @patch("whisper_to_me.menu_builder.pystray.MenuItem")
    def test_create_device_items_missing_name(self, mock_menu_item):
        """Test _create_device_items with device missing name."""
        devices = [
            {"id": 5},  # No name
            {"id": None},  # No name, no id
        ]
        current_device = None

        mock_menu_item.side_effect = [Mock(), Mock()]
        self.device_switch_handler.side_effect = [Mock(), Mock()]

        self.formatter._create_device_items(devices, current_device, False)

        calls = mock_menu_item.call_args_list
        assert "Device 5" in calls[0][0][0]
        assert "Device None" in calls[1][0][0]


class TestTrayMenuBuilder:
    """Test TrayMenuBuilder functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.builder = TrayMenuBuilder()

    @patch("whisper_to_me.menu_builder.MenuBuilder")
    def test_init(self, mock_menu_builder_class):
        """Test TrayMenuBuilder initialization."""
        mock_menu_builder = Mock()
        mock_menu_builder_class.return_value = mock_menu_builder

        builder = TrayMenuBuilder()

        assert builder.menu_builder == mock_menu_builder
        assert builder.profile_formatter is None
        assert builder.device_formatter is None

    def test_set_profile_formatter(self):
        """Test set_profile_formatter method."""
        formatter = Mock()
        self.builder.set_profile_formatter(formatter)
        assert self.builder.profile_formatter == formatter

    def test_set_device_formatter(self):
        """Test set_device_formatter method."""
        formatter = Mock()
        self.builder.set_device_formatter(formatter)
        assert self.builder.device_formatter == formatter

    def test_build_complete_menu_minimal(self):
        """Test build_complete_menu with minimal configuration."""
        mock_menu_builder = Mock()
        mock_built_menu = Mock()
        mock_menu_builder.build.return_value = mock_built_menu
        self.builder.menu_builder = mock_menu_builder

        on_quit = Mock()
        result = self.builder.build_complete_menu("default", None, on_quit)

        # Should clear, add header info, and quit option
        mock_menu_builder.clear.assert_called_once()
        mock_menu_builder.add_default_header.assert_called_once_with("Whisper-to-Me")
        mock_menu_builder.add_info_item.assert_called_with("Profile: default")
        mock_menu_builder.add_action_item.assert_called_once_with("Quit", on_quit)
        mock_menu_builder.build.assert_called_once()

        assert result == mock_built_menu

    def test_build_complete_menu_with_device(self):
        """Test build_complete_menu with device information."""
        mock_menu_builder = Mock()
        mock_built_menu = Mock()
        mock_menu_builder.build.return_value = mock_built_menu
        self.builder.menu_builder = mock_menu_builder

        current_device = {"name": "Test Audio Device"}
        on_quit = Mock()

        self.builder.build_complete_menu("work", current_device, on_quit)

        # Should include device info
        info_calls = mock_menu_builder.add_info_item.call_args_list
        assert any("Profile: work" in str(call) for call in info_calls)
        assert any("Device: Test Audio Device" in str(call) for call in info_calls)

    def test_build_complete_menu_with_long_device_name(self):
        """Test build_complete_menu with long device name."""
        mock_menu_builder = Mock()
        self.builder.menu_builder = mock_menu_builder

        current_device = {
            "name": "Very Long Audio Device Name That Should Be Truncated"
        }
        on_quit = Mock()

        self.builder.build_complete_menu("default", current_device, on_quit)

        # Check that device name is truncated
        info_calls = mock_menu_builder.add_info_item.call_args_list
        device_call = None
        for call in info_calls:
            args = call[0]
            if args and "Device:" in args[0]:
                device_call = args[0]
                break

        assert device_call is not None
        assert len(device_call) <= 38  # "Device: " + 30 chars max (27 + "...")
        assert "..." in device_call

    def test_build_complete_menu_with_profile_formatter(self):
        """Test build_complete_menu with profile formatter."""
        mock_menu_builder = Mock()
        self.builder.menu_builder = mock_menu_builder

        # Setup profile formatter
        mock_profile_formatter = Mock()
        profile_items = [Mock(), Mock()]
        mock_profile_formatter.create_profile_menu_items.return_value = profile_items
        self.builder.set_profile_formatter(mock_profile_formatter)

        on_quit = Mock()
        self.builder.build_complete_menu("default", None, on_quit)

        # Should create profile submenu
        mock_profile_formatter.create_profile_menu_items.assert_called_once()
        mock_menu_builder.add_submenu.assert_any_call("Switch Profile", profile_items)

    def test_build_complete_menu_with_device_formatter(self):
        """Test build_complete_menu with device formatter."""
        mock_menu_builder = Mock()
        self.builder.menu_builder = mock_menu_builder

        # Setup device formatter
        mock_device_formatter = Mock()
        device_items = [Mock(), Mock()]
        mock_device_formatter.create_device_menu_items.return_value = device_items
        self.builder.set_device_formatter(mock_device_formatter)

        on_quit = Mock()
        self.builder.build_complete_menu("default", None, on_quit)

        # Should create device submenu
        mock_device_formatter.create_device_menu_items.assert_called_once()
        mock_menu_builder.add_submenu.assert_any_call(
            "Select Audio Device", device_items
        )

    def test_build_complete_menu_no_profile_items(self):
        """Test build_complete_menu when profile formatter returns empty list."""
        mock_menu_builder = Mock()
        self.builder.menu_builder = mock_menu_builder

        # Setup profile formatter that returns no items
        mock_profile_formatter = Mock()
        mock_profile_formatter.create_profile_menu_items.return_value = []
        self.builder.set_profile_formatter(mock_profile_formatter)

        on_quit = Mock()
        self.builder.build_complete_menu("default", None, on_quit)

        # Should not add submenu for empty profile list
        submenu_calls = mock_menu_builder.add_submenu.call_args_list
        profile_submenu_calls = [
            call for call in submenu_calls if "Switch Profile" in str(call)
        ]
        assert len(profile_submenu_calls) == 0

    def test_build_complete_menu_no_device_items(self):
        """Test build_complete_menu when device formatter returns empty list."""
        mock_menu_builder = Mock()
        self.builder.menu_builder = mock_menu_builder

        # Setup device formatter that returns no items
        mock_device_formatter = Mock()
        mock_device_formatter.create_device_menu_items.return_value = []
        self.builder.set_device_formatter(mock_device_formatter)

        on_quit = Mock()
        self.builder.build_complete_menu("default", None, on_quit)

        # Should not add submenu for empty device list
        submenu_calls = mock_menu_builder.add_submenu.call_args_list
        device_submenu_calls = [
            call for call in submenu_calls if "Select Audio Device" in str(call)
        ]
        assert len(device_submenu_calls) == 0

    def test_build_complete_menu_full_configuration(self):
        """Test build_complete_menu with all formatters and items."""
        mock_menu_builder = Mock()
        mock_built_menu = Mock()
        mock_menu_builder.build.return_value = mock_built_menu
        self.builder.menu_builder = mock_menu_builder

        # Setup both formatters
        mock_profile_formatter = Mock()
        mock_profile_formatter.create_profile_menu_items.return_value = [Mock()]
        self.builder.set_profile_formatter(mock_profile_formatter)

        mock_device_formatter = Mock()
        mock_device_formatter.create_device_menu_items.return_value = [Mock(), Mock()]
        self.builder.set_device_formatter(mock_device_formatter)

        current_device = {"name": "Audio Device"}
        on_quit = Mock()

        result = self.builder.build_complete_menu("gaming", current_device, on_quit)

        # Check method call sequence
        mock_menu_builder.clear.assert_called_once()

        # Should have header, profile info, device info
        mock_menu_builder.add_default_header.assert_called_once_with("Whisper-to-Me")
        info_calls = mock_menu_builder.add_info_item.call_args_list
        assert len(info_calls) == 2  # Profile and device info

        # Should have separators
        separator_call_count = mock_menu_builder.add_separator.call_count
        assert separator_call_count >= 2  # At least after header and before quit

        # Should have both submenus
        submenu_calls = mock_menu_builder.add_submenu.call_args_list
        assert len(submenu_calls) == 2

        # Should have quit action
        mock_menu_builder.add_action_item.assert_called_once_with("Quit", on_quit)

        assert result == mock_built_menu
