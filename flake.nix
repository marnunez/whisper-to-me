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
            # Audio
            portaudio

            # GTK / pystray (PyGObject needs typelibs at import time)
            gtk3
            gobject-introspection
            glib
            cairo
            pango
            gdk-pixbuf
            atk

            # System tray (pystray needs AyatanaAppIndicator3 typelib)
            libayatana-appindicator

            # Wayland keystroke injection
            wtype
          ];

          # GI_TYPELIB_PATH is set automatically by gobject-introspection's setup hook,
          # but we need the typelibs from GTK and friends too.
          # LD_LIBRARY_PATH needs portaudio for sounddevice's ctypes.util.find_library.
          shellHook = ''
            export GI_TYPELIB_PATH="${pkgs.lib.makeSearchPath "lib/girepository-1.0" [
              pkgs.gtk3
              pkgs.glib
              pkgs.pango
              pkgs.gdk-pixbuf
              pkgs.atk
              pkgs.harfbuzz
              pkgs.libayatana-appindicator
            ]}''${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.portaudio
              pkgs.gtk3
              pkgs.glib
              pkgs.cairo
              pkgs.pango
              pkgs.gdk-pixbuf
            ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });
}
