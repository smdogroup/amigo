#!/usr/bin/env bash
# Build amigo in debug mode and install to the active Python environment.
# Run once (or after C++ source changes). Subsequent runs are incremental.
#
# Usage: ./build_debug.sh
#
# Cross-platform alternative (macOS / Linux / Windows):
#   Debug:    pip install -e . -Ccmake.build-type=Debug -Ccmake.build-dir=_build_debug
#   Release:  pip install -e .
#
# Note: the pip approach overwrites whichever build is currently installed,
# so only one build type is active at a time. This script keeps a separate
# _build_debug directory and copies the result into the active environment,
# which is macOS-only (requires dsymutil).

set -euo pipefail

AMIGO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}"/..)" && pwd)"
BUILD_DIR="$AMIGO_DIR/_build_debug"

# Resolve the installed .so path from the active Python environment
INSTALL_DIR=$(python -c "import amigo.amigo, os; print(os.path.dirname(amigo.amigo.__file__))")
SO_NAME=$(python -c "import amigo.amigo, os; print(os.path.basename(amigo.amigo.__file__))")

echo "==> Configuring debug build in $BUILD_DIR"
cmake -S "$AMIGO_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

echo "==> Building"
cmake --build "$BUILD_DIR" --parallel

echo "==> Generating dSYM"
dsymutil "$BUILD_DIR/$SO_NAME"

echo "==> Installing to $INSTALL_DIR"
cp "$BUILD_DIR/$SO_NAME" "$INSTALL_DIR/"
cp -r "$BUILD_DIR/$SO_NAME.dSYM" "$INSTALL_DIR/"

echo "==> Done. Debug amigo installed to $INSTALL_DIR"
echo "    Run your script under lldb with:"
echo "      lldb python -- your_script.py"
