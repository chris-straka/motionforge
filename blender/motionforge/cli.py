# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess bridge to the motionforge CLI (no bpy).

The GPL extension talks to the MIT core only through this module:
argv in, report text + output files out. All paths are passed in by the
caller (which resolves them via bpy.path); nothing here imports bpy,
so plain-python tests can exercise the arg builders.
"""

import os
import shutil
import subprocess


class CliError(Exception):
    """A CLI run that failed (message already includes stderr)."""


def find_motionforge_binary(explicit_path=""):
    """Locate the `motionforge` CLI: explicit preference, PATH, then the
    build tree relative to this extension."""
    if explicit_path and os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
        return explicit_path
    on_path = shutil.which("motionforge")
    if on_path:
        return on_path
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "..", "..", "rust", "target", "release", "motionforge"),
        os.path.join(here, "..", "..", "rust", "target", "debug", "motionforge"),
    ):
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def run_cli(binary, args, timeout=300):
    """Run the CLI; return stdout on success, raise CliError otherwise."""
    try:
        proc = subprocess.run(
            [binary] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise CliError(f"cannot run {binary}: {exc}")
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise CliError(f"motionforge exited {proc.returncode}: {stderr.strip() or stdout.strip()}")
    if stderr.strip():
        raise CliError(f"motionforge wrote to stderr: {stderr.strip()}")
    return stdout


def build_retarget_args(source, target, bonemap, output, yaw="auto", pin=True):
    args = [
        "retarget",
        "--source", source,
        "--target", target,
        "--map", bonemap,
        "--output", output,
        "--yaw", yaw,
    ]
    args.append("--pin" if pin else "--no-pin")
    return args


def build_stylize_args(input_path, output, params):
    """params: dict with the StylizeParams keys (chain_factors a list of
    (substr, factor) tuples)."""
    args = ["stylize", "--input", input_path, "--output", output]
    for key in (
        "exaggeration",
        "angle_threshold",
        "min_spacing",
        "hold",
        "anticipation",
        "anticipation_frames",
        "overshoot",
    ):
        args += ["--" + key.replace("_", "-"), repr(params[key])]
    for substr, factor in params.get("chain_factors", []):
        args += ["--chain", f"{substr}:{factor!r}"]
    return args


def build_physics_args(command, input_path, output, params):
    """command: 'physics-check' or 'physics-fix'. params: dict with the
    PhysicsParams keys (feet a list of names)."""
    args = [command, "--input", input_path]
    if output is not None:
        args += ["--output", output]
    args += ["--root", params["root"], "--feet", ",".join(params["feet"])]
    for key in (
        "contact_margin",
        "foot_radius",
        "balance_margin",
        "min_air_frames",
        "accel_limit",
        "turn_deg",
        "min_turn_speed",
    ):
        args += ["--" + key.replace("_", "-"), repr(params[key])]
    if command == "physics-fix":
        for key in ("blend_frames", "smooth_sigma", "smooth_pad"):
            args += ["--" + key.replace("_", "-"), repr(params[key])]
        if not params.get("fix_ballistic", True):
            args.append("--no-ballistic")
        if not params.get("fix_momentum", True):
            args.append("--no-momentum")
    return args


def build_autopose_args(model, effectors, output):
    return ["autopose", "--model", model, "--effectors", effectors, "--output", output]


def parse_chain_factors(text):
    """Parse 'substr:factor, ...' into [(substr, factor)]. Empty -> []."""
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            raise ValueError(f"chain factors must be 'substr:factor', got '{chunk}'")
        substr, _, factor = chunk.partition(":")
        substr, factor = substr.strip(), factor.strip()
        if not substr:
            raise ValueError(f"chain factors need a bone substring, got '{chunk}'")
        try:
            value = float(factor)
        except ValueError:
            raise ValueError(f"chain factor must be a number, got '{chunk}'")
        out.append((substr, value))
    return out


def parse_feet(text):
    """Parse 'a, b, ...' into [names]. Empty -> ValueError."""
    names = [n.strip() for n in (text or "").split(",") if n.strip()]
    if not names:
        raise ValueError("feet needs at least one bone name")
    return names
