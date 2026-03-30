{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            # Build tools
            pkg-config
            gobject-introspection  # g-ir-scanner, setup hooks for GI_TYPELIB_PATH
          ];

          buildInputs = with pkgs; [
            # Audio (sounddevice needs libportaudio)
            portaudio

            # C build deps (evdev extension needs kernel headers)
            linuxHeaders

            # Cairo/GTK (pycairo + pygobject + pystray)
            cairo
            glib
            gtk3
            pango
            gdk-pixbuf
            atk
            harfbuzz

            # System tray (pystray needs AyatanaAppIndicator3 typelib)
            libayatana-appindicator

            # Wayland keystroke injection
            wtype
          ];

          shellHook = ''
            # GI_TYPELIB_PATH for PyGObject (pystray, etc.)
            export GI_TYPELIB_PATH="${pkgs.lib.makeSearchPath "lib/girepository-1.0" [
              pkgs.gtk3
              pkgs.glib
              pkgs.pango
              pkgs.gdk-pixbuf
              pkgs.atk
              pkgs.harfbuzz
              pkgs.libayatana-appindicator
            ]}''${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"

            # portaudio for sounddevice's ctypes.util.find_library
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.portaudio
            ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });
}
