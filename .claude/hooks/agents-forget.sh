#!/usr/bin/env bash
# Compaction can summarise away notes that agents-context.sh already fed in, while its per-session
# stamps still claim they are in context. Clearing the stamps makes the next edit re-inject them.
set -euo pipefail
session=$(cat | jq -r '.session_id // "nosession"')
rm -rf "${TMPDIR:-/tmp}/claude-agents-$session"
