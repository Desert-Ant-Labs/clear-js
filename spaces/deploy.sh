#!/bin/bash
# Stage the clear-web demo + the Spaces README into a temp directory, then
# upload to a HuggingFace Space via `hf upload`. Model files are NOT bundled;
# the demo fetches them directly from huggingface.co/detail-co/clear at runtime
# (works because the Space's COEP is `credentialless` and HF Hub returns a
# CORS response that echoes the requesting Space's origin).
#
# Usage:
#   clear-web/spaces/deploy.sh <space-id>
#
# Example:
#   source clear-web/../.env  # for HF_TOKEN
#   clear-web/spaces/deploy.sh desert-ant-labs/clear-demo
#
# Prereqs:
#   - hf CLI installed, write-scoped token via HF_TOKEN env or `hf auth login`.
#   - The Space must already exist (create it once at
#     https://huggingface.co/new-space, pick "Static" as the SDK).
set -e

SPACE_ID="${1:-}"
if [ -z "$SPACE_ID" ]; then
    echo "usage: $0 <space-id> (e.g. desert-ant-labs/clear-demo)" >&2
    exit 1
fi

WEB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap "rm -rf $STAGE" EXIT

echo "→ staging files in $STAGE"
cp "$WEB_DIR/index.html" "$STAGE/"
cp "$WEB_DIR/main.js"    "$STAGE/"
cp "$WEB_DIR/clear.css"  "$STAGE/"
cp -R "$WEB_DIR/lib"     "$STAGE/"
cp "$WEB_DIR/spaces/README.md" "$STAGE/"

echo "→ uploading to https://huggingface.co/spaces/$SPACE_ID"
hf upload "$SPACE_ID" "$STAGE" . --repo-type=space --commit-message "Update Clear browser demo"

echo
echo "✓ deployed → https://huggingface.co/spaces/$SPACE_ID"
