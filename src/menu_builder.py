"""
Menu Builder Module

Provides classes for building complex tray icon menus with cleaner separation of concerns.
Extracted from the tray icon to reduce complexity and improve maintainability.
"""

from typing import Optional, Callable, List, Dict
import pystray


class MenuBuilder:
    """
    Base class for building pystray menus with common functionality.

    Features:
    - Menu item creation utilities
    - Separator handling
    - Handler function generation
    - Consistent menu structure
    """

    def __init__(self):
        """Initialize the menu builder."""
        self.menu_items: List[pystray.MenuItem] = []

    def add_header(self, text: str, enabled: bool = False) -> None:
        """
        Add a header item to the menu.

        Args:
            text: Header text
            enabled: Whether the header is clickable
        """
        self.menu_items.append(pystray.MenuItem(text, None, enabled=enabled))

    def add_default_header(self, text: str) -> None:
        """
        Add a default (highlighted) header item.

        Args:
            text: Header text
        """
        self.menu_items.append(
            pystray.MenuItem(text, None, default=True, enabled=False)
        )

    def add_info_item(self, text: str) -> None:
        """
        Add an informational (non-clickable) item.

        Args:
            text: Information text
        """
        self.menu_items.append(pystray.MenuItem(text, None, enabled=False))

    def add_separator(self) -> None:
        """Add a menu separator."""
        self.menu_items.append(pystray.Menu.SEPARATOR)

    def add_action_item(self, text: str, handler: Callable) -> None:
        """
        Add a clickable action item.

        Args:
            text: Menu item text
            handler: Function to call when clicked
        """
        self.menu_items.append(pystray.MenuItem(text, handler))

    def add_submenu(self, text: str, submenu_items: List[pystray.MenuItem]) -> None:
        """
        Add a submenu with nested items.

        Args:
            text: Submenu label
            submenu_items: List of menu items for the submenu
        """
        self.menu_items.append(pystray.MenuItem(text, pystray.Menu(*submenu_items)))

    def build(self) -> pystray.Menu:
        """
        Build and return the complete menu.

        Returns:
            pystray.Menu object
        """
        return pystray.Menu(*self.menu_items)

    def clear(self) -> None:
        """Clear all menu items."""
        self.menu_items.clear()


class ProfileMenuFormatter:
    """
    Formatter for profile-related menu items.

    Features:
    - Profile switching menu generation
    - Current profile highlighting
    - Profile validation
    """

    def __init__(
        self,
        get_profiles: Callable[[], List[str]],
        get_current_profile: Callable[[], str],
        profile_switch_handler: Callable[[str], Callable],
    ):
        """
        Initialize the profile menu formatter.

        Args:
            get_profiles: Function to get available profiles
            get_current_profile: Function to get current profile
            profile_switch_handler: Function to create profile switch handlers
        """
        self.get_profiles = get_profiles
        self.get_current_profile = get_current_profile
        self.profile_switch_handler = profile_switch_handler

    def create_profile_menu_items(self) -> List[pystray.MenuItem]:
        """
        Create menu items for profile switching.

        Returns:
            List of menu items for profile switching
        """
        profiles = self.get_profiles()
        current_profile = self.get_current_profile()

        if len(profiles) <= 1:
            return []

        profile_items = []
        for profile in profiles:
            marker = "●" if profile == current_profile else "○"
            display_name = f"{marker} {profile}"

            profile_items.append(
                pystray.MenuItem(display_name, self.profile_switch_handler(profile))
            )

        return profile_items


class DeviceMenuFormatter:
    """
    Formatter for audio device-related menu items.

    Features:
    - Device menu generation grouped by host API
    - Current device highlighting
    - Device name truncation
    - Host API organization
    """

    def __init__(
        self,
        get_devices: Callable[[], List[Dict]],
        get_current_device: Callable[[], Optional[Dict]],
        device_switch_handler: Callable[[int], Callable],
    ):
        """
        Initialize the device menu formatter.

        Args:
            get_devices: Function to get available devices
            get_current_device: Function to get current device
            device_switch_handler: Function to create device switch handlers
        """
        self.get_devices = get_devices
        self.get_current_device = get_current_device
        self.device_switch_handler = device_switch_handler

    def create_device_menu_items(self) -> List[pystray.MenuItem]:
        """
        Create menu items for device switching.

        Returns:
            List of menu items for device switching
        """
        devices = self.get_devices()
        current_device = self.get_current_device()

        if not devices or len(devices) <= 1:
            return []

        # Group devices by host API
        devices_by_hostapi = self._group_devices_by_hostapi(devices)
        device_items = []

        # If only one host API, don't create submenus
        if len(devices_by_hostapi) == 1:
            hostapi_name = list(devices_by_hostapi.keys())[0]
            device_items.extend(
                self._create_device_items(
                    devices_by_hostapi[hostapi_name], current_device, False
                )
            )
        else:
            # Create submenus for each host API
            for hostapi_name in sorted(devices_by_hostapi.keys()):
                hostapi_devices = devices_by_hostapi[hostapi_name]
                hostapi_items = self._create_device_items(
                    hostapi_devices, current_device, True
                )

                device_items.append(
                    pystray.MenuItem(hostapi_name, pystray.Menu(*hostapi_items))
                )

        return device_items

    def _group_devices_by_hostapi(self, devices: List[Dict]) -> Dict[str, List[Dict]]:
        """Group devices by their host API."""
        grouped = {}
        for device in devices:
            hostapi_name = device.get("hostapi_name", "Unknown")
            if hostapi_name not in grouped:
                grouped[hostapi_name] = []
            grouped[hostapi_name].append(device)
        return grouped

    def _create_device_items(
        self, devices: List[Dict], current_device: Optional[Dict], is_nested: bool
    ) -> List[pystray.MenuItem]:
        """Create menu items for a list of devices."""
        items = []
        max_name_length = 35 if is_nested else 40

        for device in devices:
            device_name = device.get("name", f"Device {device.get('id', '?')}")

            # Truncate long device names
            if len(device_name) > max_name_length:
                device_name = device_name[: max_name_length - 3] + "..."

            # Mark current device
            is_current = current_device and device.get("id") == current_device.get("id")
            display_name = f"✓ {device_name}" if is_current else device_name

            items.append(
                pystray.MenuItem(
                    display_name, self.device_switch_handler(device.get("id"))
                )
            )

        return items


class TrayMenuBuilder:
    """
    Main tray menu builder that orchestrates all menu sections.

    Combines header information, profile switching, device switching,
    and action items into a cohesive menu structure.
    """

    def __init__(self):
        """Initialize the tray menu builder."""
        self.menu_builder = MenuBuilder()
        self.profile_formatter: Optional[ProfileMenuFormatter] = None
        self.device_formatter: Optional[DeviceMenuFormatter] = None

    def set_profile_formatter(self, formatter: ProfileMenuFormatter) -> None:
        """Set the profile menu formatter."""
        self.profile_formatter = formatter

    def set_device_formatter(self, formatter: DeviceMenuFormatter) -> None:
        """Set the device menu formatter."""
        self.device_formatter = formatter

    def build_complete_menu(
        self, current_profile: str, current_device: Optional[Dict], on_quit: Callable
    ) -> pystray.Menu:
        """
        Build the complete tray menu with all sections.

        Args:
            current_profile: Name of current active profile
            current_device: Current audio device information
            on_quit: Quit handler function

        Returns:
            Complete pystray.Menu
        """
        self.menu_builder.clear()

        # Add header
        self.menu_builder.add_default_header("Whisper-to-Me")
        self.menu_builder.add_info_item(f"Profile: {current_profile}")

        # Add current device info if available
        if current_device:
            device_name = current_device.get("name", "Unknown")
            if len(device_name) > 30:  # Truncate long device names
                device_name = device_name[:27] + "..."
            self.menu_builder.add_info_item(f"Device: {device_name}")

        self.menu_builder.add_separator()

        # Add profile switching if multiple profiles exist
        if self.profile_formatter:
            profile_items = self.profile_formatter.create_profile_menu_items()
            if profile_items:
                self.menu_builder.add_submenu("Switch Profile", profile_items)
                self.menu_builder.add_separator()

        # Add device switching if multiple devices exist
        if self.device_formatter:
            device_items = self.device_formatter.create_device_menu_items()
            if device_items:
                self.menu_builder.add_submenu("Select Audio Device", device_items)
                self.menu_builder.add_separator()

        # Add quit option
        self.menu_builder.add_action_item("Quit", on_quit)

        return self.menu_builder.build()
