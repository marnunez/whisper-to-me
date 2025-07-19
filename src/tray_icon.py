"""
System Tray Icon Module

Provides a system tray icon interface for Whisper-to-Me with status indicators
and menu controls.
"""

import pystray
from PIL import Image, ImageDraw, ImageOps
import threading
import os
from typing import Optional, Callable


class TrayIcon:
    """
    System tray icon handler for Whisper-to-Me.
    
    Features:
    - Visual recording status indicator
    - Right-click menu with options
    - Cross-platform support
    """
    
    def __init__(self, on_quit: Optional[Callable] = None):
        """
        Initialize the tray icon.
        
        Args:
            on_quit: Callback function to call when quit is selected
        """
        self.icon: Optional[pystray.Icon] = None
        self.on_quit_callback = on_quit
        self.is_recording = False
        self._running = False
        
    def create_image(self, recording: bool = False) -> Image.Image:
        """
        Create the tray icon image.
        
        Args:
            recording: Whether currently recording
            
        Returns:
            PIL Image for the tray icon
        """
        # Get the path to the icon
        icon_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'icons')
        icon_path = os.path.join(icon_dir, 'mic-32.png')
        
        # Use fallback if icon doesn't exist
        if not os.path.exists(icon_path):
            return self._create_fallback_icon(recording)
        
        # Load the icon
        try:
            icon = Image.open(icon_path).convert('RGBA')
            
            # Create a clean transparent background
            result = Image.new('RGBA', icon.size, (0, 0, 0, 0))
            
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
                draw.ellipse([x, y, x + dot_size, y + dot_size], fill=(255, 255, 255, 255))
                draw.ellipse([x+1, y+1, x + dot_size-1, y + dot_size-1], fill=(255, 0, 0, 255))
                
            return result
                
        except Exception as e:
            print(f"Error loading icon: {e}")
            return self._create_fallback_icon(recording)
    
    def _create_fallback_icon(self, recording: bool = False) -> Image.Image:
        """Create a simple fallback icon if the PNG file is not found."""
        size = 32
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Simple circle icon
        color = (220, 38, 38, 255) if recording else (107, 114, 128, 255)
        draw.ellipse([4, 4, size-4, size-4], fill=color)
        
        # Inner circle
        draw.ellipse([8, 8, size-8, size-8], fill=(255, 255, 255, 100))
        
        return image
    
    def update_icon(self, recording: bool):
        """
        Update the tray icon to reflect recording status.
        
        Args:
            recording: Whether currently recording
        """
        self.is_recording = recording
        if self.icon:
            self.icon.icon = self.create_image(recording)
            
    def on_activate(self, icon, item):
        """Handle menu item activation."""
        pass
    
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
        return pystray.Menu(
            pystray.MenuItem("Whisper-to-Me", self.on_activate, default=True, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Status: {'Recording' if self.is_recording else 'Ready'}",
                self.on_activate,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.on_quit)
        )
    
    def run(self):
        """Run the system tray icon."""
        self._running = True
        self.icon = pystray.Icon(
            "whisper-to-me",
            self.create_image(),
            "Whisper-to-Me - Press and hold trigger key to record",
            menu=self.create_menu()
        )
        
        # Run the icon
        self.icon.run()
    
    def start(self):
        """Start the tray icon in a separate thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop the tray icon."""
        self._running = False
        if self.icon:
            self.icon.stop()