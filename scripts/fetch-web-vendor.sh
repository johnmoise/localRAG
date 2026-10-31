#!/usr/bin/env bash
# Vendored front-end libraries (committed to web/vendor/ so builds work offline).
# marked (MIT), DOMPurify (Apache-2.0/MPL-2.0), PDF.js (Apache-2.0).
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/web/vendor"
CDN="https://cdn.jsdelivr.net/npm"
MARKED=15.0.12
DOMPURIFY=3.2.6
PDFJS=4.10.38

mkdir -p "$DEST"
curl -fsSL "$CDN/marked@$MARKED/marked.min.js" -o "$DEST/marked.min.js"
curl -fsSL "$CDN/dompurify@$DOMPURIFY/dist/purify.min.js" -o "$DEST/purify.min.js"
curl -fsSL "$CDN/pdfjs-dist@$PDFJS/build/pdf.min.mjs" -o "$DEST/pdf.min.mjs"
curl -fsSL "$CDN/pdfjs-dist@$PDFJS/build/pdf.worker.min.mjs" -o "$DEST/pdf.worker.min.mjs"
printf 'marked %s\ndompurify %s\npdfjs-dist %s\n' "$MARKED" "$DOMPURIFY" "$PDFJS" > "$DEST/VERSIONS"
ls -l "$DEST"
