"""
System Tray Icon Module

Provides a system tray icon interface for Whisper-to-Me with status indicators
and menu controls.
"""

import pystray
from PIL import Image, ImageDraw
import threading
import os
from typing import Optional, Callable, List
from whisper_to_me.logger import get_logger


class TrayIcon:
    """
    System tray icon handler for Whisper-to-Me.

    Features:
    - Visual recording status indicator
    - Right-click menu with options
    - Cross-platform support
    """

    def __init__(
        self,
        on_quit: Optional[Callable] = None,
        on_profile_change: Optional[Callable[[str], None]] = None,
        get_profiles: Optional[Callable[[], List[str]]] = None,
        get_current_profile: Optional[Callable[[], str]] = None,
        on_device_change: Optional[Callable[[int], None]] = None,
        get_devices: Optional[Callable[[], List[dict]]] = None,
        get_current_device: Optional[Callable[[], Optional[dict]]] = None,
    ):
        """
        Initialize the tray icon.

        Args:
            on_quit: Callback function to call when quit is selected
            on_profile_change: Callback function to call when profile is changed
            get_profiles: Function to get list of available profiles
            get_current_profile: Function to get current active profile
            on_device_change: Callback function to call when audio device is changed
            get_devices: Function to get list of available audio devices
            get_current_device: Function to get current audio device info
        """
        self.icon: Optional[pystray.Icon] = None
        self.on_quit_callback = on_quit
        self.on_profile_change_callback = on_profile_change
        self.get_profiles_callback = get_profiles
        self.get_current_profile_callback = get_current_profile
        self.on_device_change_callback = on_device_change
        self.get_devices_callback = get_devices
        self.get_current_device_callback = get_current_device
        self.is_recording = False
        self._running = False
        self.current_profile = "default"
        self.logger = get_logger()

    def create_image(self, recording: bool = False) -> Image.Image:
        """
        Create the tray icon image.

        Args:
            recording: Whether currently recording

        Returns:
            PIL Image for the tray icon
        """
        # Get the path to the icon
        icon_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "icons"
        )
        icon_path = os.path.join(icon_dir, "mic-32.png")

        # Use fallback if icon doesn't exist
        if not os.path.exists(icon_path):
            return self._create_fallback_icon(recording)

        # Load the icon
        try:
            icon = Image.open(icon_path).convert("RGBA")

            # Create a clean transparent background
            result = Image.new("RGBA", icon.size, (0, 0, 0, 0))

            # Get the pixel data
            pixels = icon.load()
            result_pixels = result.load()

            for y in range(icon.size[1]):
                for x in range(icon.size[0]):
                    r, g, b, a = pixels[x, y]

                    # If pixel has some alpha (not completely transparent)
                    if a > 10:  # Small threshold to avoid artifacts
                        if recording:
                            # Red color for recording
                            result_pixels[x, y] = (220, 38, 38, a)
                        else:
                            # Dark gray for idle (better visibility)
                            result_pixels[x, y] = (60, 60, 60, a)
                    else:
                        # Keep transparent
                        result_pixels[x, y] = (0, 0, 0, 0)

            # Add recording indicator dot if recording
            if recording:
                draw = ImageDraw.Draw(result)
                dot_size = 6
                x, y = result.size[0] - dot_size - 1, 1
                draw.ellipse(
                    [x, y, x + dot_size, y + dot_size], fill=(255, 255, 255, 255)
                )
                draw.ellipse(
                    [x + 1, y + 1, x + dot_size - 1, y + dot_size - 1],
                    fill=(255, 0, 0, 255),
                )

            return result

        except Exception as e:
            self.logger.error(f"Error loading icon: {e}", "ui")
            return self._create_fallback_icon(recording)

    def _create_fallback_icon(self, recording: bool = False) -> Image.Image:
        """Create a simple fallback icon if the PNG file is not found."""
        size = 32
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Simple circle icon
        color = (220, 38, 38, 255) if recording else (107, 114, 128, 255)
        draw.ellipse([4, 4, size - 4, size - 4], fill=color)

        # Inner circle
        draw.ellipse([8, 8, size - 8, size - 8], fill=(255, 255, 255, 100))

        return image

    def update_icon(self, recording: bool):
        """
        Update the tray icon to reflect recording status.

        Args:
            recording: Whether currently recording
        """
        self.is_recording = recording
        if self.icon:
            try:
                self.icon.icon = self.create_image(recording)
            except Exception as e:
                self.logger.error(f"Error updating icon: {e}", "ui")

    def update_profile(self, profile_name: str):
        """Update the current profile and refresh the menu."""
        self.current_profile = profile_name
        if self.icon:
            try:
                # Refresh the menu
                self.icon.menu = self.create_menu()
                # Update tooltip
                self.icon.title = f"Whisper-to-Me (Profile: {profile_name})"
            except Exception as e:
                self.logger.error(f"Error updating profile: {e}", "ui")

    def refresh_menu(self):
        """Manually refresh the tray menu."""
        if self.icon:
            try:
                self.icon.menu = self.create_menu()
            except Exception as e:
                self.logger.error(f"Error refreshing menu: {e}", "ui")

    def on_activate(self, icon, item):
        """Handle menu item activation (left-click on icon)."""
        pass

    def on_profile_select(self, icon, item, profile_name: str):
        """Handle profile selection from menu."""
        if self.on_profile_change_callback:
            self.on_profile_change_callback(profile_name)
        self.update_profile(profile_name)

    def on_device_select(self, icon, item, device_id: int):
        """Handle audio device selection from menu."""
        if self.on_device_change_callback:
            self.on_device_change_callback(device_id)

    def on_quit(self, icon, item):
        """Handle quit menu item."""
        self.stop()
        if self.on_quit_callback:
            self.on_quit_callback()

    def create_menu(self) -> pystray.Menu:
        """
        Create the right-click menu for the tray icon.

        Returns:
            Menu object with options
        """
        # Get current profile for display
        current_profile = self.current_profile
        if self.get_current_profile_callback:
            current_profile = self.get_current_profile_callback()
            self.current_profile = current_profile

        # Get available profiles for menu
        profiles = []
        if self.get_profiles_callback:
            profiles = self.get_profiles_callback()

        # Get available devices for menu
        devices = []
        current_device = None
        if self.get_devices_callback:
            devices = self.get_devices_callback()
        if self.get_current_device_callback:
            current_device = self.get_current_device_callback()

        menu_items = [
            pystray.MenuItem(
                "Whisper-to-Me", self.on_activate, default=True, enabled=False
            ),
            pystray.MenuItem(f"Profile: {current_profile}", None, enabled=False),
        ]

        # Add current device info if available
        if current_device:
            device_name = current_device.get("name", "Unknown")
            if len(device_name) > 30:  # Truncate long device names
                device_name = device_name[:27] + "..."
            menu_items.append(
                pystray.MenuItem(f"Device: {device_name}", None, enabled=False)
            )

        menu_items.append(pystray.Menu.SEPARATOR)

        # Add profile switching options if multiple profiles exist
        if len(profiles) > 1:
            profile_menu_items = []
            for profile in profiles:
                profile_menu_items.append(
                    pystray.MenuItem(
                        profile, self._create_profile_switch_handler(profile)
                    )
                )
            menu_items.append(
                pystray.MenuItem("Switch Profile", pystray.Menu(*profile_menu_items))
            )
            menu_items.append(pystray.Menu.SEPARATOR)

        # Add device switching options if devices are available
        if devices and len(devices) > 1:
            # Group devices by host API
            devices_by_hostapi = {}
            for device in devices:
                hostapi_name = device.get("hostapi_name", "Unknown")
                if hostapi_name not in devices_by_hostapi:
                    devices_by_hostapi[hostapi_name] = []
                devices_by_hostapi[hostapi_name].append(device)

            # Create device submenu grouped by host API
            device_menu_items = []

            # Sort host APIs for consistent ordering
            for hostapi_name in sorted(devices_by_hostapi.keys()):
                hostapi_devices = devices_by_hostapi[hostapi_name]

                # If only one host API, don't create submenus
                if len(devices_by_hostapi) == 1:
                    for device in hostapi_devices:
                        device_name = device.get(
                            "name", f"Device {device.get('id', '?')}"
                        )
                        if len(device_name) > 40:
                            device_name = device_name[:37] + "..."

                        is_current = current_device and device.get(
                            "id"
                        ) == current_device.get("id")
                        display_name = f"✓ {device_name}" if is_current else device_name

                        device_menu_items.append(
                            pystray.MenuItem(
                                display_name,
                                self._create_device_switch_handler(device.get("id")),
                            )
                        )
                else:
                    # Create submenu for each host API
                    hostapi_menu_items = []
                    for device in hostapi_devices:
                        device_name = device.get(
                            "name", f"Device {device.get('id', '?')}"
                        )
                        if len(device_name) > 35:  # Shorter for nested menus
                            device_name = device_name[:32] + "..."

                        is_current = current_device and device.get(
                            "id"
                        ) == current_device.get("id")
                        display_name = f"✓ {device_name}" if is_current else device_name

                        hostapi_menu_items.append(
                            pystray.MenuItem(
                                display_name,
                                self._create_device_switch_handler(device.get("id")),
                            )
                        )

                    device_menu_items.append(
                        pystray.MenuItem(
                            hostapi_name, pystray.Menu(*hostapi_menu_items)
                        )
                    )

            menu_items.append(
                pystray.MenuItem(
                    "Select Audio Device", pystray.Menu(*device_menu_items)
                )
            )
            menu_items.append(pystray.Menu.SEPARATOR)

        menu_items.append(pystray.MenuItem("Quit", self.on_quit))

        return pystray.Menu(*menu_items)

    def _create_profile_switch_handler(self, profile_name: str):
        """Create a handler for profile switching."""

        def handler(icon, item):
            self.on_profile_select(icon, item, profile_name)

        return handler

    def _create_device_switch_handler(self, device_id: int):
        """Create a handler for device switching."""

        def handler(icon, item):
            self.on_device_select(icon, item, device_id)

        return handler

    def run(self):
        """Run the system tray icon."""
        self._running = True

        # Get current profile for tooltip
        current_profile = self.current_profile
        if self.get_current_profile_callback:
            current_profile = self.get_current_profile_callback()
            self.current_profile = current_profile

        try:
            # Create the menu
            menu = self.create_menu()

            self.icon = pystray.Icon(
                "whisper-to-me",
                self.create_image(),
                f"Whisper-to-Me (Profile: {current_profile})",
                menu=menu,
            )

            # Run the icon
            self.icon.run()

        except Exception as e:
            self.logger.error(f"Error creating/running tray icon: {e}", "ui")
            import traceback

            traceback.print_exc()

    def start(self):
        """Start the tray icon in a separate thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def stop(self):
        """Stop the tray icon."""
        self._running = False
        if self.icon:
            self.icon.stop()
