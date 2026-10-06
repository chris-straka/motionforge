// motionforge CLI: retarget, stylize, physics-check/fix, autopose, clip-info,
// and the GLB rig commands + genforge adapter (glb_cmds.rs).
//
// Reports go to stdout (deterministic, golden-pinned); warnings, errors,
// and --time diagnostics go to stderr. Files are written only to
// --output. Exit codes: 0 ok, 1 usage error, 2 input error, 3
// processing error.

use motion_core::autopose::{infer, parse_effectors, parse_weights};
use motion_core::clip::{emit_clip, emit_skeleton, parse_clip, Clip, Frame, Pose};
use motion_core::limits::{check_limits, parse_limits};
use motion_core::math::Vec3;
use motion_core::physics::{physics_fix, PhysicsParams};
use motion_core::retarget::{parse_bonemap, parse_target_skeleton, retarget, YawMode};
use motion_core::stylize::StylizeParams;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

mod glb_cmds;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = "motionforge {VERSION} — game-animation sidekick for HLL\n\
\n\
usage: motionforge <command> [flags]\n\
\n\
commands:\n\
  retarget       transfer a source clip onto a target skeleton\n\
  stylize        realistic clip in, snappy game motion out\n\
  physics-check  report balance/ballistic/momentum errors (no output)\n\
  physics-frame  single-frame balance snapshot as JSON (3D overlay)\n\
  physics-fix    fix ballistic + momentum errors via root curves\n\
  autopose       predict a full pose from a few effector joints\n\
  clip-info      validate a clip and print its inventory\n\
  standardize    rig GLB -> HLL skeleton (DEF-* names, canonical parents)\n\
  animate        retarget clip GLBs onto a rigged GLB (+ contact pass)\n\
  grip           find a weapon clip set's grip, write it into the hand socket\n\
  pose-test      range-of-motion poses as animations (+ rendered sheet)\n\
  fixture-glb    procedural skinned test humanoid GLB\n\
  adapter        genforge character-chain adapter (adapter --help)\n\
\n\
retarget flags:\n\
  --source <clip> --target <skeleton|clip> --map <bonemap> --output <clip>\n\
  --yaw auto|flip|none (default auto)  --pin|--no-pin (default --pin)\n\
\n\
stylize flags:\n\
  --input <clip> --output <clip>\n\
  --exaggeration <0..4> (default 1.35)  --chain <substr:factor> (repeatable)\n\
  --angle-threshold <rad> (default 0.15)  --min-spacing <n> (default 4)\n\
  --hold <n> (default 2)  --anticipation <0..1> (default 0.25)\n\
  --anticipation-frames <n> (default 3)  --overshoot <0..2> (default 0.6)\n\
  --keys-out <keys> (per-bone key frames sidecar for thinned import)\n\
\n\
physics-frame flags:\n\
  --input <clip> --frame <n> --root <bone> --feet <a,b,...>\n\
  (contact/foot/balance margins as physics-check; prints overlay JSON)\n\
\n\
physics-check flags:\n\
  --input <clip> --root <bone> --feet <a,b,...>\n\
  --contact-margin <m> (default 0.03)  --foot-radius <m> (default 0.12)\n\
  --balance-margin <m> (default 0.05)  --min-air-frames <n> (default 4)\n\
  --accel-limit <m/s2> (default 25)  --turn-deg <deg> (default 40)\n\
  --min-turn-speed <m/s> (default 0.5)\n\
\n\
physics-fix flags: physics-check flags plus:\n\
  --output <clip>  --blend-frames <n> (default 2)\n\
  --smooth-sigma <frames> (default 1)  --smooth-pad <n> (default 2)\n\
  --no-ballistic  --no-momentum  --no-balance\n\
  --max-lean-deg <deg> (default 8, total lean cap per frame)\n\
\n\
autopose flags:\n\
  --model <weights> --effectors <effectors> --output <clip>\n\
  [--limits <limits>] (report joint-limit violations, exit 0)\n\
\n\
clip-info flags:\n\
  --input <clip> [--emit-skeleton <skeleton>]\n\
\n\
standardize flags:\n\
  --input <glb> --output <glb> [--class humanoid|quadruped|custom] [--report <json>]\n\
\n\
animate flags:\n\
  --input <glb> --output <glb> --clips <glb|folder> (repeatable) [--fps 30]\n\
  --pick <name=new,label.glb:name=new,...> (choose, rename and order clips)\n\
  --no-contact (skip the contact pass)  --weapon R|L|none (default R)\n\
  --weapon-length <heights> (default 0.75)  --weapon-clips <a,b> (name parts)\n\
  --contact-ramp <s> (default 0.15)  --proxies <json> (write collision proxies)\n\
\n\
grip flags:\n\
  --input <glb with weapon clips> [--side R|L] [--weapon-length <heights>]\n\
  [--weapon-clips <a,b>] [--output <glb with the socket turned>]\n\
\n\
pose-test flags:\n\
  --input <glb> --output <posed glb> [--sheet <png>] [--clips ...] [--blender <path>]\n\
\n\
fixture-glb flags:\n\
  --output <glb> [--naming mixamo|plain|def] [--twisted] [--walk]\n\
\n\
global: --help/-h, --version/-v, --time (wall ms on stderr)\n";

fn help_text() -> String {
    HELP.replace("{VERSION}", VERSION)
}

struct Cmd {
    opts: HashMap<String, Vec<String>>,
    bools: HashMap<String, bool>,
}

impl Cmd {
    fn get(&self, key: &str) -> Option<&str> {
        self.opts
            .get(key)
            .and_then(|v| v.first().map(|s| s.as_str()))
    }

    fn all(&self, key: &str) -> &[String] {
        self.opts.get(key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn flag(&self, key: &str) -> bool {
        self.bools.get(key).copied().unwrap_or(false)
    }

    fn required(&self, key: &str) -> Result<&str, (i32, String)> {
        self.get(key)
            .ok_or_else(|| (1, format!("missing required --{}", key)))
    }
}

fn is_bool_flag(key: &str) -> bool {
    matches!(
        key,
        "pin"
            | "no-pin"
            | "no-ballistic"
            | "no-momentum"
            | "no-balance"
            | "twisted"
            | "no-contact"
            | "walk"
            | "time"
            | "help"
            | "h"
            | "version"
            | "v"
    )
}

/// Parse `argv` (minus argv[0]) into a command name + flags. Values bind
/// `--key value` or `--key=value`; known bool flags take no value.
fn parse_argv(argv: &[String]) -> Result<(String, Cmd), (i32, String)> {
    if argv.is_empty() {
        return Err((1, "no command; try --help".to_string()));
    }
    let mut words = argv.to_vec();
    let cmd = words.remove(0);
    if cmd == "--help" || cmd == "-h" || cmd == "help" {
        return Ok((
            "help".to_string(),
            Cmd {
                opts: HashMap::new(),
                bools: HashMap::new(),
            },
        ));
    }
    if cmd == "--version" || cmd == "-v" || cmd == "version" {
        return Ok((
            "version".to_string(),
            Cmd {
                opts: HashMap::new(),
                bools: HashMap::new(),
            },
        ));
    }
    if cmd.starts_with('-') {
        return Err((
            1,
            format!("expected a command, found '{}'; try --help", cmd),
        ));
    }
    let mut opts: HashMap<String, Vec<String>> = HashMap::new();
    let mut bools: HashMap<String, bool> = HashMap::new();
    let mut i = 0;
    while i < words.len() {
        let w = &words[i];
        if !w.starts_with("--") || w.len() < 3 {
            return Err((
                1,
                format!("unexpected '{}'; flags look like --key value", w),
            ));
        }
        let (key, inline) = match w.find('=') {
            Some(eq) => (w[2..eq].to_string(), Some(w[eq + 1..].to_string())),
            None => (w[2..].to_string(), None),
        };
        if key.is_empty() {
            return Err((1, "empty flag name".to_string()));
        }
        if is_bool_flag(&key) {
            if inline.is_some() {
                return Err((1, format!("--{} takes no value", key)));
            }
            bools.insert(key, true);
            i += 1;
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => {
                i += 1;
                if i >= words.len() {
                    return Err((1, format!("--{} needs a value", key)));
                }
                words[i].clone()
            }
        };
        opts.entry(key).or_default().push(value);
        i += 1;
    }
    Ok((cmd, Cmd { opts, bools }))
}

fn read_file(path: &str) -> Result<String, (i32, String)> {
    std::fs::read_to_string(path).map_err(|e| (2, format!("cannot read {}: {}", path, e)))
}

fn write_file(path: &str, text: &str) -> Result<(), (i32, String)> {
    std::fs::write(path, text).map_err(|e| (3, format!("cannot write {}: {}", path, e)))
}

fn parse_f64(cmd: &Cmd, key: &str, default: f64) -> Result<f64, (i32, String)> {
    match cmd.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| (1, format!("--{} must be a number, got '{}'", key, v)))
            .and_then(|f| {
                if f.is_finite() {
                    Ok(f)
                } else {
                    Err((1, format!("--{} must be finite", key)))
                }
            }),
    }
}

fn parse_usize(cmd: &Cmd, key: &str, default: usize) -> Result<usize, (i32, String)> {
    match cmd.get(key) {
        None => Ok(default),
        Some(v) => v.parse::<usize>().map_err(|_| {
            (
                1,
                format!("--{} must be a non-negative integer, got '{}'", key, v),
            )
        }),
    }
}

fn cmd_retarget(cmd: &Cmd) -> Result<String, (i32, String)> {
    let source_text = read_file(cmd.required("source")?)?;
    let target_text = read_file(cmd.required("target")?)?;
    let map_text = read_file(cmd.required("map")?)?;
    let output = cmd.required("output")?.to_string();
    let source = parse_clip(&source_text).map_err(|e| (2, format!("source: {}", e)))?;
    let target = parse_target_skeleton(&target_text).map_err(|e| (2, format!("target: {}", e)))?;
    let map = parse_bonemap(&map_text).map_err(|e| (2, format!("map: {}", e)))?;
    let yaw = match cmd.get("yaw").unwrap_or("auto") {
        "auto" => YawMode::Auto,
        "flip" => YawMode::Flip,
        "none" => YawMode::None,
        other => return Err((1, format!("--yaw must be auto|flip|none, got '{}'", other))),
    };
    if cmd.flag("pin") && cmd.flag("no-pin") {
        return Err((1, "--pin and --no-pin conflict".to_string()));
    }
    let pin = !cmd.flag("no-pin");
    let (out, report) = retarget(&source, &target, &map, yaw, pin).map_err(|e| (3, e))?;
    let text = emit_clip(&out).map_err(|e| (3, e))?;
    write_file(&output, &text)?;

    let mut r = String::new();
    r.push_str("=== motionforge retarget ===\n");
    r.push_str(&format!(
        "mapped: {} pairs ({} skipped)\n",
        report.mapped,
        report.skipped.len()
    ));
    for s in &report.skipped {
        r.push_str(&format!("  skip: {}\n", s));
    }
    r.push_str(&format!("chain roots: {}\n", report.chain_roots.join(", ")));
    r.push_str(&format!(
        "stride scale: {:.4}{}\n",
        report.stride_scale,
        if report.stride_defaulted {
            " (defaulted, no stride pair)"
        } else {
            ""
        }
    ));
    r.push_str(&format!(
        "yaw: {}\n",
        if report.yaw_flip { "flip" } else { "none" }
    ));
    if let Some(note) = &report.yaw_note {
        r.push_str(&format!("yaw note: {}\n", note));
    }
    r.push_str("foot slide before:\n");
    if report.slide_before.is_empty() {
        r.push_str("  (no feet in map)\n");
    }
    for s in &report.slide_before {
        r.push_str(&format!(
            "  {}: {} stance frames, mean {:.4} m/s, max {:.4} m/s (overall max {:.4})\n",
            s.name, s.stance_frames, s.stance_mean_m_s, s.stance_max_m_s, s.overall_max_m_s
        ));
    }
    r.push_str("foot slide after:\n");
    if report.slide_after.is_empty() {
        r.push_str("  (no feet in map)\n");
    }
    for s in &report.slide_after {
        r.push_str(&format!(
            "  {}: {} stance frames, mean {:.4} m/s, max {:.4} m/s (overall max {:.4})\n",
            s.name, s.stance_frames, s.stance_mean_m_s, s.stance_max_m_s, s.overall_max_m_s
        ));
    }
    if pin {
        r.push_str(&format!(
            "pin: {} intervals, total drift {:.4} m\n",
            report.pin_intervals, report.pin_drift_m
        ));
    }
    r.push_str(&format!(
        "root travel: {:.4} m, max swing: {:.4} rad\n",
        report.root_travel_m, report.max_swing_rad
    ));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

fn cmd_stylize(cmd: &Cmd) -> Result<String, (i32, String)> {
    let input_text = read_file(cmd.required("input")?)?;
    let output = cmd.required("output")?.to_string();
    let clip = parse_clip(&input_text).map_err(|e| (2, format!("input: {}", e)))?;
    let mut chain_factors = Vec::new();
    for raw in cmd.all("chain") {
        let (substr, factor) = raw
            .rsplit_once(':')
            .ok_or_else(|| (1, format!("--chain must be substr:factor, got '{}'", raw)))?;
        let factor: f64 = factor
            .parse()
            .map_err(|_| (1, format!("--chain factor must be a number, got '{}'", raw)))?;
        chain_factors.push((substr.to_string(), factor));
    }
    let params = StylizeParams {
        exaggeration: parse_f64(cmd, "exaggeration", 1.35)?,
        chain_factors,
        angle_threshold: parse_f64(cmd, "angle-threshold", 0.15)?,
        min_spacing: parse_usize(cmd, "min-spacing", 4)?,
        hold: parse_usize(cmd, "hold", 2)?,
        anticipation: parse_f64(cmd, "anticipation", 0.25)?,
        anticipation_frames: parse_usize(cmd, "anticipation-frames", 3)?,
        overshoot: parse_f64(cmd, "overshoot", 0.6)?,
    };
    let (out, report) = motion_core::stylize::stylize(&clip, &params).map_err(|e| (3, e))?;
    let text = emit_clip(&out).map_err(|e| (3, e))?;
    write_file(&output, &text)?;
    if let Some(keys_path) = cmd.get("keys-out") {
        write_file(
            keys_path,
            &motion_core::stylize::emit_keys(&clip.skeleton, &report),
        )?;
    }

    let mut r = String::new();
    r.push_str("=== motionforge stylize ===\n");
    r.push_str(&format!(
        "frames: {}, keys kept: {} {:?}\n",
        clip.frames.len(),
        report.keys_union.len(),
        report.keys_union
    ));
    for (bone, count) in &report.busy_bones {
        r.push_str(&format!("  {}: {} keys\n", bone, count));
    }
    r.push_str(&format!(
        "anticipation: {} added, {} skipped (no room)\n",
        report.counters_added, report.counters_skipped
    ));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

fn physics_params(cmd: &Cmd, for_fix: bool) -> Result<PhysicsParams, (i32, String)> {
    let feet_raw = cmd.required("feet")?;
    let feet: Vec<String> = feet_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if feet.is_empty() {
        return Err((1, "--feet needs at least one bone".to_string()));
    }
    Ok(PhysicsParams {
        root: cmd.required("root")?.to_string(),
        feet,
        contact_margin: parse_f64(cmd, "contact-margin", 0.03)?,
        foot_radius: parse_f64(cmd, "foot-radius", 0.12)?,
        balance_margin: parse_f64(cmd, "balance-margin", 0.05)?,
        min_air_frames: parse_usize(cmd, "min-air-frames", 4)?,
        blend_frames: parse_usize(cmd, "blend-frames", 2)?,
        accel_limit: parse_f64(cmd, "accel-limit", 25.0)?,
        turn_deg: parse_f64(cmd, "turn-deg", 40.0)?,
        min_turn_speed: parse_f64(cmd, "min-turn-speed", 0.5)?,
        smooth_sigma: parse_f64(cmd, "smooth-sigma", 1.0)?,
        smooth_pad: parse_usize(cmd, "smooth-pad", 2)?,
        fix_ballistic: for_fix && !cmd.flag("no-ballistic"),
        fix_momentum: for_fix && !cmd.flag("no-momentum"),
        fix_balance: for_fix && !cmd.flag("no-balance"),
        max_lean_deg: parse_f64(cmd, "max-lean-deg", 8.0)?,
    })
}

fn physics_report_text(title: &str, report: &motion_core::physics::PhysicsReport) -> String {
    let mut r = String::new();
    r.push_str(&format!("=== motionforge {} ===\n", title));
    r.push_str(&format!(
        "contact frames: {}, airborne: {}\n",
        report.contact_frames, report.airborne_frames
    ));
    r.push_str(&format!(
        "balance: {} violations, mean excursion {:+.4} m\n",
        report.balance_violations.len(),
        report.mean_excursion_m
    ));
    for v in report.balance_violations.iter().take(8) {
        r.push_str(&format!(
            "  frame {}: excursion {:+.4} m (nudge {:+.4}, {:+.4})\n",
            v.frame, v.excursion_m, v.nudge.0, v.nudge.1
        ));
    }
    if report.balance_violations.len() > 8 {
        r.push_str(&format!(
            "  ... and {}\n",
            report.balance_violations.len() - 8
        ));
    }
    r.push_str(&format!(
        "ballistic: {} phases\n",
        report.ballistic_phases.len()
    ));
    for p in &report.ballistic_phases {
        r.push_str(&format!(
            "  frames {}..{}: residual {:.4} m\n",
            p.start, p.end, p.residual_m
        ));
    }
    r.push_str(&format!(
        "momentum: max accel {:.2} m/s2 ({} flags {:?}), max turn {:.1} deg ({} flags {:?})\n",
        report.max_accel_m_s2,
        report.accel_flags.len(),
        report.accel_flags,
        report.max_turn_deg,
        report.turn_flags.len(),
        report.turn_flags
    ));
    r
}

fn cmd_physics_frame(cmd: &Cmd) -> Result<String, (i32, String)> {
    let input_text = read_file(cmd.required("input")?)?;
    let clip = parse_clip(&input_text).map_err(|e| (2, format!("input: {}", e)))?;
    let frame = parse_usize(cmd, "frame", usize::MAX)?;
    if frame == usize::MAX && cmd.get("frame").is_none() {
        return Err((1, "missing required --frame".to_string()));
    }
    let mut params = physics_params(cmd, false)?;
    params.fix_ballistic = false;
    params.fix_momentum = false;
    let snap = motion_core::physics::frame_physics(&clip, &params, frame).map_err(|e| (3, e))?;
    motion_core::physics::emit_frame_physics(&snap).map_err(|e| (3, e))
}

fn cmd_physics_check(cmd: &Cmd) -> Result<String, (i32, String)> {
    let input_text = read_file(cmd.required("input")?)?;
    let clip = parse_clip(&input_text).map_err(|e| (2, format!("input: {}", e)))?;
    let mut params = physics_params(cmd, false)?;
    params.fix_ballistic = false;
    params.fix_momentum = false;
    let (_, report) = physics_fix(&clip, &params).map_err(|e| (3, e))?;
    Ok(physics_report_text("physics-check", &report))
}

fn cmd_physics_fix(cmd: &Cmd) -> Result<String, (i32, String)> {
    let input_text = read_file(cmd.required("input")?)?;
    let output = cmd.required("output")?.to_string();
    let clip = parse_clip(&input_text).map_err(|e| (2, format!("input: {}", e)))?;
    let params = physics_params(cmd, true)?;
    let (out, report) = physics_fix(&clip, &params).map_err(|e| (3, e))?;
    let text = emit_clip(&out).map_err(|e| (3, e))?;
    write_file(&output, &text)?;
    let mut r = physics_report_text("physics-fix", &report);
    r.push_str(&format!(
        "fixed: {} ballistic phases, {} momentum frames, {} balance frames \
         (worst {:+.4} -> {:+.4} m)\n",
        report.fixed_ballistic,
        report.fixed_momentum_frames,
        report.fixed_balance_frames,
        report.balance_worst_before_m,
        report.balance_worst_after_m
    ));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

fn cmd_autopose(cmd: &Cmd) -> Result<String, (i32, String)> {
    let model_text = read_file(cmd.required("model")?)?;
    let eff_text = read_file(cmd.required("effectors")?)?;
    let output = cmd.required("output")?.to_string();
    let weights = parse_weights(&model_text).map_err(|e| (2, format!("model: {}", e)))?;
    let input = parse_effectors(&eff_text).map_err(|e| (2, format!("effectors: {}", e)))?;
    let quats = infer(&weights, &input).map_err(|e| (3, e))?;
    let clip = Clip {
        fps: 30.0,
        skeleton: input.skeleton.clone(),
        frames: vec![Frame {
            poses: quats
                .iter()
                .map(|q| Pose {
                    loc: Vec3::ZERO,
                    quat: *q,
                })
                .collect(),
        }],
    };
    let text = emit_clip(&clip).map_err(|e| (3, e))?;
    write_file(&output, &text)?;
    let mut r = String::new();
    r.push_str("=== motionforge autopose ===\n");
    r.push_str(&format!(
        "effectors: {}, bones: {}\n",
        input.effectors.len(),
        weights.n_bones()
    ));
    if let Some(path) = cmd.get("limits") {
        let lim_text = read_file(path)?;
        let limits = parse_limits(&lim_text).map_err(|e| (2, format!("limits: {}", e)))?;
        let bones: Vec<String> = input
            .skeleton
            .bones
            .iter()
            .map(|b| b.name.clone())
            .collect();
        let rep = check_limits(&limits, &bones, &[quats.clone()]).map_err(|e| (3, e))?;
        r.push_str(&format!(
            "limits: {} violations, {} unmatched\n",
            rep.violations.len(),
            rep.unmatched.len()
        ));
        for v in &rep.violations {
            r.push_str(&format!(
                "  frame {} {}: {:.2} deg > {:.2} deg\n",
                v.frame, v.bone, v.angle_deg, v.max_deg
            ));
        }
        if !rep.unmatched.is_empty() {
            r.push_str(&format!("  unmatched: {}\n", rep.unmatched.join(", ")));
        }
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

fn cmd_clip_info(cmd: &Cmd) -> Result<String, (i32, String)> {
    let input_text = read_file(cmd.required("input")?)?;
    let clip = parse_clip(&input_text).map_err(|e| (2, format!("input: {}", e)))?;
    let mut r = String::new();
    r.push_str("=== motionforge clip-info ===\n");
    r.push_str(&format!(
        "bones: {}, frames: {}, fps: {}, duration: {:.3} s\n",
        clip.skeleton.len(),
        clip.frames.len(),
        clip.fps,
        clip.frames.len() as f64 / clip.fps
    ));
    for bone in &clip.skeleton.bones {
        r.push_str(&format!(
            "  {} (parent: {}, len {:.4} m)\n",
            bone.name,
            bone.parent
                .map(|p| clip.skeleton.bones[p].name.as_str())
                .unwrap_or("-"),
            clip.skeleton
                .bone_length(clip.skeleton.index(&bone.name).unwrap())
        ));
    }
    if let Some(path) = cmd.get("emit-skeleton") {
        let text = emit_skeleton(&clip.skeleton).map_err(|e| (3, e))?;
        write_file(path, &text)?;
        r.push_str(&format!("skeleton: {}\n", path));
    }
    Ok(r)
}

fn real_main(argv: &[String]) -> (i32, Option<String>) {
    let want_time = argv.iter().any(|a| a == "--time");
    let argv: Vec<String> = argv
        .iter()
        .filter(|a| a.as_str() != "--time")
        .cloned()
        .collect();
    let started = Instant::now();
    let result = match parse_argv(&argv) {
        Ok((cmd, args)) => match cmd.as_str() {
            "help" => Ok(help_text()),
            "version" => Ok(format!("motionforge {}\n", VERSION)),
            "retarget" => cmd_retarget(&args),
            "stylize" => cmd_stylize(&args),
            "physics-check" => cmd_physics_check(&args),
            "physics-frame" => cmd_physics_frame(&args),
            "physics-fix" => cmd_physics_fix(&args),
            "autopose" => cmd_autopose(&args),
            "clip-info" => cmd_clip_info(&args),
            "standardize" => glb_cmds::cmd_standardize(&args),
            "animate" => glb_cmds::cmd_animate(&args),
            "grip" => glb_cmds::cmd_grip(&args),
            "pose-test" => glb_cmds::cmd_pose_test(&args),
            "fixture-glb" => glb_cmds::cmd_fixture_glb(&args),
            other => Err((1, format!("unknown command '{}'; try --help", other))),
        },
        Err(e) => Err(e),
    };
    if want_time {
        eprintln!("wall: {:.3} ms", started.elapsed().as_secs_f64() * 1000.0);
    }
    match result {
        Ok(report) => (0, Some(report)),
        Err((code, msg)) => {
            eprintln!("motionforge: {}", msg);
            (code, None)
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("adapter") {
        let code = glb_cmds::adapter_main(&argv[1..]);
        std::io::stdout().flush().ok();
        std::process::exit(code);
    }
    let (code, report) = real_main(&argv);
    if let Some(text) = report {
        print!("{}", text);
    }
    // process::exit does not flush stdout; flush explicitly first.
    std::io::stdout().flush().ok();
    std::process::exit(code);
}
