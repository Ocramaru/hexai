#!/usr/bin/env bash
# Convention: any directory may hold .agents/ with context.md, learned.md and todo.md.
# On a file edit, walk that file's directory up to the repo root and feed in every .agents
# found, outermost first so the most specific notes land last.
#
# Names are matched case insensitively, though lowercase is the convention. Each file is fed in
# once per session; re-sending it on every edit would spend context to say the same thing.
set -euo pipefail

payload=$(cat)
path=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
[ -n "$path" ] || exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
dir=$(cd "$(dirname "$path")" 2>/dev/null && pwd) || exit 0
case "$dir" in "$root"*) ;; *) exit 0 ;; esac   # ignore anything outside the repo

session=$(printf '%s' "$payload" | jq -r '.session_id // "nosession"')
seen="${TMPDIR:-/tmp}/claude-agents-$session"
mkdir -p "$seen"

chain=()
while [ "$dir" != "/" ]; do
  chain=("$dir" "${chain[@]}")                  # prepend, so we end up root-first
  [ "$dir" = "$root" ] && break
  dir=$(dirname "$dir")
done

blob=""
for d in "${chain[@]}"; do
  agents=$(find "$d" -maxdepth 1 -iname ".agents" -type d 2>/dev/null | head -1)
  [ -n "$agents" ] || continue
  for want in context learned todo; do          # what it is, what we found, what is open
    f=$(find "$agents" -maxdepth 1 -iname "$want.md" -type f 2>/dev/null | head -1)
    [ -n "$f" ] || continue
    stamp="$seen/$(printf '%s' "$f" | md5sum | cut -c1-16)"
    [ -e "$stamp" ] && continue                 # already in context this session
    : > "$stamp"
    blob="$blob--- ${f#"$root"/} ---
$(cat "$f")

"
  done
done

[ -n "$blob" ] || exit 0
jq -n --arg text "Notes from .agents on the path to this file. Read before changing behaviour.

$blob" '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $text}}'
