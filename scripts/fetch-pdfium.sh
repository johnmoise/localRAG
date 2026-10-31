#!/usr/bin/env bash
# Download the prebuilt PDFium library for Apple Silicon into third_party/pdfium/.
# The version matches the newest PDFium API that pdfium-render supports (feature pdfium_7881).
set -euo pipefail

VERSION="${PDFIUM_VERSION:-7881}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/pdfium"
URL="https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${VERSION}/pdfium-mac-arm64.tgz"

mkdir -p "$DEST"
echo "Downloading PDFium chromium/${VERSION} → ${DEST}"
curl -fsSL "$URL" | tar -xz -C "$DEST"
echo "$VERSION" > "$DEST/VERSION"
ls -l "$DEST/lib/libpdfium.dylib"
