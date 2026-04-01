#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <extension.so> [reference.sass] [kernel_regex]" >&2
  exit 1
fi

SO_PATH="$1"
REFERENCE_PATH="${2:-}"
KERNEL_REGEX="${3:-causal_machine_(forward|backward).*tiled.*kernel}"

if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "cuobjdump not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$SO_PATH" ]]; then
  echo "extension not found: $SO_PATH" >&2
  exit 1
fi

TMP_OUTPUT="$(mktemp)"
cleanup() {
  rm -f "$TMP_OUTPUT"
}
trap cleanup EXIT

cuobjdump -sass "$SO_PATH" | grep -E "$KERNEL_REGEX" -A200 -B2 >"$TMP_OUTPUT" || true

if [[ ! -s "$TMP_OUTPUT" ]]; then
  echo "no matching kernels found for regex: $KERNEL_REGEX" >&2
  exit 1
fi

if [[ -n "$REFERENCE_PATH" ]]; then
  if [[ ! -f "$REFERENCE_PATH" ]]; then
    echo "reference not found: $REFERENCE_PATH" >&2
    exit 1
  fi
  diff -u "$REFERENCE_PATH" "$TMP_OUTPUT"
else
  cat "$TMP_OUTPUT"
fi
