motionforge licensing split
============================

- `rust/` (core + CLI), `python/`, `tools/`, `tests/`, `docs/`:
  MIT (see `LICENSE-MIT`).
- `blender/` (the Blender extension): GPL-3.0-or-later, like all
  Blender add-ons. The GPL extension talks to the MIT core only as a
  subprocess over clip JSON files — never linked, never imported —
  so the engine stays MIT-licensed.

`rust/core/src/detmath.rs` ports FreeBSD msun routines (Copyright (C)
1993, 2004 Sun Microsystems, Inc.; "Permission to use, copy, modify,
and distribute this software is freely granted, provided that this
notice is preserved") via musl and rust-lang/libm (MIT). The notice is
kept in the file header. Source code only, no crate dependency.

Rigs, GLBs, and animations produced *with* these tools are the
owner's content and are unaffected by either license.
