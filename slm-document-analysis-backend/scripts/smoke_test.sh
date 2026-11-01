#!/usr/bin/env bash
# Manual end-to-end check against a running server (default http://127.0.0.1:8000).
# Usage: scripts/smoke_test.sh path/to/file.pdf "Your question?"
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
PDF="${1:?usage: smoke_test.sh file.pdf \"question\"}"
QUESTION="${2:-What is this document about?}"

echo "--- GET /health"
curl -sS "$BASE/health"; echo

echo "--- POST /answer (PDF + question in one request)"
curl -sS -F "file=@${PDF};type=application/pdf" -F "question=${QUESTION}" "$BASE/answer"; echo
