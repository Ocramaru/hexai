---
allowed-tools: Bash(find:*), Bash(cat:*), Bash(ls:*), Read
description: Load the .agents notes for a directory, or list what exists
---

## Available notes

!`find . -type d -name .agents -not -path "./.venv/*" -not -path "./.git/*" 2>/dev/null | while read d; do echo "$d:"; find "$d" -maxdepth 1 -iname "*.md" -type f | sed 's|.*/|    |'; done`

## Your task

The argument is a path, a directory, or a topic: **$ARGUMENTS**

Read the `.agents` notes that apply to it and say what they change about how you would work there.

- With an argument, load the `.agents` on the path from that file or directory up to the repo root, outermost first. A topic rather than a path means pick the `.agents` directory whose name matches.
- Without one, list what exists above and ask which is wanted rather than loading everything.

By convention each `.agents` holds `context.md` (why the design is what it is), `learned.md` (what experiments actually showed, and the traps that already cost a run) and `todo.md` (open tasks). Names may be any case. A directory may hold only some of them.

Do not summarise a document back at length. Say what is settled, what is still open, and anything that contradicts what you were about to do.
