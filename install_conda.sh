#!/usr/bin/env bash
set -euo pipefail

# 1. Where to install Miniconda
INSTALL_DIR="$HOME/miniconda3"

# 2. Detect OS and architecture for the right installer
OS="$(uname)"
ARCH="$(uname -m)"
case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)    PKG="Miniconda3-latest-Linux-x86_64.sh" ;;
      aarch64)   PKG="Miniconda3-latest-Linux-aarch64.sh" ;;
      *)         echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64)    PKG="Miniconda3-latest-MacOSX-x86_64.sh" ;;
      arm64)     PKG="Miniconda3-latest-MacOSX-arm64.sh" ;;
      *)         echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

URL="https://repo.anaconda.com/miniconda/${PKG}"

# 3. Download installer
echo "Downloading $PKG..."
curl -fsSL "$URL" -o "$PKG"

# 4. Run silent installer
echo "Installing to $INSTALL_DIR..."
bash "$PKG" -b -p "$INSTALL_DIR"

# 5. Clean up
rm "$PKG"

# 6. Initialize the right shell
SHELL_NAME="$(basename "${SHELL:-/bin/bash}")"
echo "Initializing conda for $SHELL_NAME..."
"$INSTALL_DIR/bin/conda" init "$SHELL_NAME"

# 7. Reload profile so that conda is on PATH right away
PROFILE="$HOME/.bashrc"
if [[ "$SHELL_NAME" == "zsh" ]]; then
  PROFILE="$HOME/.zshrc"
fi
echo "Sourcing $PROFILE..."
# shellcheck disable=SC1090
source "$PROFILE"

# 8. Update conda itself
echo "Updating base conda..."
conda update -n base -c defaults conda -y

# 9. Final check
echo
echo "✅ Conda is installed at: $(which conda)"
echo "✅ Conda version: $(conda --version)"
