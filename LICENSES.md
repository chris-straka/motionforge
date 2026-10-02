motionforge licensing split
============================

- `rust/` (core + CLI), `python/`, `tools/`, `tests/`, `docs/`:
  MIT (see `LICENSE-MIT`).
- `blender/` (the Blender extension): GPL-3.0-or-later, like all
  Blender add-ons. The GPL extension talks to the MIT core only as a
  subprocess over clip JSON files — never linked, never imported —
  so the engine stays MIT-licensed.

Rigs, GLBs, and animations produced *with* these tools are the
owner's content and are unaffected by either license.
