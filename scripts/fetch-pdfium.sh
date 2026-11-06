#!/usr/bin/env bash
# Download the prebuilt PDFium library (bblanchon/pdfium-binaries) into third_party/pdfium/.
# The version matches the newest PDFium API that pdfium-render supports (feature pdfium_7881).
#
# The platform defaults to this machine (macOS on Apple Silicon → mac-arm64, Windows under Git
# Bash → win-x64). Set PDFIUM_PLATFORM=win-x64 on a Mac to fetch pdfium.dll for a
# cross-compiled Windows build; both builds can live side by side in third_party/pdfium/
# (lib/libpdfium.dylib and bin/pdfium.dll).
set -euo pipefail

VERSION="${PDFIUM_VERSION:-7881}"
if [[ -z "${PDFIUM_PLATFORM:-}" ]]; then
  case "$(uname -s)" in
    Darwin) PDFIUM_PLATFORM=mac-arm64 ;;
    MINGW*|MSYS*|CYGWIN*) PDFIUM_PLATFORM=win-x64 ;;
    *) echo "unsupported platform $(uname -s); set PDFIUM_PLATFORM (e.g. mac-arm64, win-x64)" >&2; exit 1 ;;
  esac
fi
case "$PDFIUM_PLATFORM" in
  win-*) LIB=bin/pdfium.dll ;;
  mac-*) LIB=lib/libpdfium.dylib ;;
  *) LIB=lib/libpdfium.so ;;
esac

DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/pdfium"
URL="https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${VERSION}/pdfium-${PDFIUM_PLATFORM}.tgz"

mkdir -p "$DEST"
echo "Downloading PDFium chromium/${VERSION} (${PDFIUM_PLATFORM}) → ${DEST}"
curl -fsSL "$URL" | tar -xz -C "$DEST"
echo "$VERSION" > "$DEST/VERSION"
ls -l "$DEST/$LIB"
