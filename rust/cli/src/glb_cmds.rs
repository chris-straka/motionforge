// GLB rig commands (standardize, animate, pose-test, fixture-glb) and the
// genforge adapter contract.
//
// genforge contract (genforge docs/phase-7.md, same schema as the
// weightforge adapter): `motionforge adapter <stage> IN.glb OUT.glb
// RESULT.json [flags]` writes RESULT.json = {"ok": bool, "outputs":
// [paths relative to RESULT.json's folder], "tool": "motionforge", ...}.
// Exit 0 = ok, 1 = ran but the stage failed (result written, ok false),
// 2 = error (no result written, so the chain stops instead of guessing).

use motion_core::animate::{parse_pick, retarget_clips, retarget_with, round};
use motion_core::contact::Options as ContactOptions;
use motion_core::fixture::{self, Naming};
use motion_core::glb::Document;
use motion_core::json::{emit_pretty, Json};
use motion_core::posetest;
use motion_core::standardize::{standardize, Outcome};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{parse_argv, Cmd};

const POSE_SHEET_PY: &str = include_str!("../../../blender/tools/pose_sheet.py");
const VERSION: &str = env!("CARGO_PKG_VERSION");

type CmdResult = Result<String, (i32, String)>;

fn load(path: &str) -> Result<Document, (i32, String)> {
    Document::read(path).map_err(|e| (2, e))
}

fn save(doc: &Document, path: &str) -> Result<(), (i32, String)> {
    if let Some(dir) = Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir)
            .map_err(|e| (3, format!("cannot create {}: {}", dir.display(), e)))?;
    }
    doc.write(path).map_err(|e| (3, e))
}

fn class_of(cmd: &Cmd) -> Result<String, (i32, String)> {
    let class = cmd.get("class").unwrap_or("humanoid");
    if !matches!(class, "humanoid" | "quadruped" | "custom") {
        return Err((
            1,
            format!("--class must be humanoid|quadruped|custom, got '{}'", class),
        ));
    }
    Ok(class.to_string())
}

/// Clip sources: each `--clips` value is a .glb or a folder of them.
fn clip_sources(cmd: &Cmd) -> Result<Vec<(String, Document)>, (i32, String)> {
    let mut files: Vec<PathBuf> = Vec::new();
    for value in cmd.all("clips") {
        let p = PathBuf::from(value);
        if p.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&p)
                .map_err(|e| (2, format!("cannot list {}: {}", p.display(), e)))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    f.extension()
                        .map(|x| x.eq_ignore_ascii_case("glb"))
                        .unwrap_or(false)
                })
                .collect();
            found.sort();
            if found.is_empty() {
                return Err((2, format!("no .glb clips in {}", p.display())));
            }
            files.extend(found);
        } else if p.is_file() {
            files.push(p);
        } else {
            return Err((2, format!("clip source not found: {}", value)));
        }
    }
    let mut out = Vec::new();
    for f in files {
        let label = f
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        out.push((label, load(&f.to_string_lossy())?));
    }
    Ok(out)
}

fn pretty(json: &Json) -> Result<String, (i32, String)> {
    emit_pretty(json).map_err(|e| (3, e))
}

pub fn cmd_standardize(cmd: &Cmd) -> CmdResult {
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let class = class_of(cmd)?;
    let doc = load(input)?;
    match standardize(&doc, &class).map_err(|e| (2, e))? {
        Outcome::Done(out, report) => {
            save(&out, output)?;
            if let Some(r) = cmd.get("report") {
                std::fs::write(r, pretty(&report.to_json())?).map_err(|e| (3, e.to_string()))?;
            }
            Ok(format!(
                "=== motionforge standardize ===\nclass: {}\njoints: {} -> {}\nrenamed: {}\nreparented: {}\nmerged: {}\nanimations dropped: {}\noutput: {}\n",
                class,
                report.joints_in,
                report.joints_out,
                report.renamed.len(),
                report.reparented.len(),
                report.merged.len(),
                report.animations_dropped,
                output
            ))
        }
        Outcome::Refused(_, why) => Err((3, why)),
    }
}

/// Contact-pass settings from flags (on unless `--no-contact`).
fn contact_opts(cmd: &Cmd) -> Result<Option<ContactOptions>, (i32, String)> {
    if cmd.flag("no-contact") {
        return Ok(None);
    }
    let mut o = ContactOptions::default();
    if let Some(w) = cmd.get("weapon") {
        o.weapon_side = match w {
            "R" | "r" => Some("R".into()),
            "L" | "l" => Some("L".into()),
            "none" => None,
            other => return Err((1, format!("--weapon must be R|L|none, got '{}'", other))),
        };
    }
    o.weapon_length = crate::parse_f64(cmd, "weapon-length", o.weapon_length)?;
    if let Some(c) = cmd.get("weapon-clips") {
        o.weapon_clips = c
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
    }
    o.ramp = crate::parse_f64(cmd, "contact-ramp", o.ramp)?;
    if !(0.0..=2.0).contains(&o.ramp) || !(0.0..=2.0).contains(&o.weapon_length) {
        return Err((
            1,
            "--contact-ramp and --weapon-length must be in 0..2".into(),
        ));
    }
    Ok(Some(o))
}

fn contact_line(c: &motion_core::contact::ClipContact) -> String {
    let mut parts = Vec::new();
    for a in &c.arms {
        if a.frames_before == 0 && a.mesh_frames_before == 0 {
            continue;
        }
        parts.push(format!(
            "{}: proxies {} frames ({:.1} cm) -> {}; mesh {} frames inside ({:.1} cm) -> {} ({:.1} cm); wrist moved up to {:.1} cm in {} frames",
            a.side,
            a.frames_before,
            a.max_depth_before * 100.0,
            a.frames_after,
            a.mesh_frames_before,
            a.mesh_depth_before * 100.0,
            a.mesh_frames_after,
            a.mesh_depth_after * 100.0,
            a.max_shift * 100.0,
            a.frames_changed
        ));
    }
    if parts.is_empty() {
        "  contact: clean\n".into()
    } else {
        format!("  contact {}\n", parts.join("; "))
    }
}

pub fn cmd_animate(cmd: &Cmd) -> CmdResult {
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let fps = crate::parse_f64(cmd, "fps", 30.0)?;
    if !(1.0..=240.0).contains(&fps) {
        return Err((1, "--fps must be in 1..240".into()));
    }
    let sources = clip_sources(cmd)?;
    if sources.is_empty() {
        return Err((1, "missing required --clips".into()));
    }
    let doc = load(input)?;
    let pick = match cmd.get("pick") {
        Some(spec) => Some(parse_pick(spec).map_err(|e| (1, e))?),
        None => None,
    };
    let contact = contact_opts(cmd)?;
    let pin = !cmd.flag("no-pin");
    let (out, reports) = retarget_with(&doc, &sources, fps, pick.as_ref(), contact.as_ref(), pin)
        .map_err(|e| (3, e))?;
    // Optional passes on the retargeted clips, in order: the body
    // stylizer (on the clips `--stylize` names, `attacks` = the attack
    // clips), then `clean` (floor, lifts, plant lock).
    let mut extra = String::new();
    let out = match cmd.get("stylize") {
        Some(which) => {
            use motion_core::stylize_body::{stylize_glb, BodyParams, Mode};
            let names: Vec<String> = if which == "attacks" {
                motion_core::rig::Animation::load_all(&out)
                    .map_err(|e| (3, e))?
                    .iter()
                    .filter(|a| {
                        motion_core::motioncheck::is_attack(
                            &a.name,
                            &motion_core::motioncheck::ATTACK_WORDS
                                .iter()
                                .map(|w| w.to_string())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .map(|a| a.name.clone())
                    .collect()
            } else {
                which
                    .split(',')
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty())
                    .collect()
            };
            if names.is_empty() {
                out
            } else {
                let (o, reps) =
                    stylize_glb(&out, &BodyParams::default(), Mode::Body, &names, None, true)
                        .map_err(|e| (3, e))?;
                for c in &reps {
                    extra.push_str(&format!("  stylized {}: keys {:?}\n", c.name, c.keys));
                }
                o
            }
        }
        None => out,
    };
    let out = if cmd.flag("clean") {
        use motion_core::clean::{clean_glb, CleanOptions};
        let (o, reps) = clean_glb(&out, &CleanOptions::default(), &[], None).map_err(|e| (3, e))?;
        for c in &reps {
            extra.push_str(&format!(
                "  clean {}: ground {:+.1} cm, hips up to {:.1} cm, {} plants locked\n",
                c.name,
                c.ground_shift_m * 100.0,
                c.hips_lift_max_m * 100.0,
                c.plants_locked
            ));
        }
        o
    } else {
        out
    };
    if let Some(path) = cmd.get("proxies") {
        let rig = motion_core::rig::Rig::from_doc(&doc).map_err(|e| (3, e))?;
        let body =
            motion_core::contact::Body::build(&doc, &rig, &contact.clone().unwrap_or_default())
                .map_err(|e| (3, e))?
                .ok_or((3, "no collision proxies for this rig".to_string()))?;
        std::fs::write(path, pretty(&body.to_json(&rig))?).map_err(|e| (3, e.to_string()))?;
    }
    save(&out, output)?;
    let mut r = String::from("=== motionforge animate ===\n");
    for c in &reports {
        r.push_str(&format!(
            "clip {} ({}): {} frames, {} bones, max error {} deg, root travel {} m\n",
            c.name,
            c.source,
            c.frames,
            c.bones_driven,
            round(c.max_error_deg, 6),
            round(c.root_travel_m, 4)
        ));
        if let Some(p) = &c.pin {
            r.push_str(&format!(
                "  pin: {} plants, {} frames, ankle moved up to {:.1} cm\n",
                p.plants,
                p.frames_changed,
                p.max_shift * 100.0
            ));
        }
        if let Some(ct) = &c.contact {
            r.push_str(&contact_line(ct));
        }
    }
    r.push_str(&extra);
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// `motionforge grip`: find the blade direction a GLB's own weapon clips
/// were animated for (the direction that keeps the blade out of the body)
/// and write it into its `Socket_Hand_<side>`.
pub fn cmd_grip(cmd: &Cmd) -> CmdResult {
    use motion_core::contact::grip_search;
    use motion_core::rig::{Animation, Rig};
    let input = cmd.required("input")?;
    let doc = load(input)?;
    let side = cmd.get("side").unwrap_or("R");
    let opts = contact_opts(cmd)?.unwrap_or_default();
    let rig = Rig::from_doc(&doc).map_err(|e| (3, e))?;
    let map = motion_core::humanoid::map_rig(&rig);
    let hand = map
        .node_of(&format!("DEF-hand.{}", side))
        .ok_or((3, "no hand".to_string()))?;
    let socket = rig.children[hand]
        .iter()
        .copied()
        .find(|&c| rig.names[c] == format!("Socket_Hand_{}", side))
        .ok_or((
            3,
            format!("no Socket_Hand_{} (run standardize first)", side),
        ))?;
    let origin = rig.rest_world[socket].t;
    let mut clips = Vec::new();
    let mut names = Vec::new();
    for anim in Animation::load_all(&doc).map_err(|e| (3, e))? {
        if !opts.weapon_in(&anim.name) {
            continue;
        }
        let n = ((anim.duration * 15.0).round() as usize).max(1) + 1;
        clips.push(
            (0..n)
                .map(|f| anim.sample(&rig, anim.duration * f as f64 / (n - 1).max(1) as f64))
                .collect::<Vec<_>>(),
        );
        names.push(anim.name.clone());
    }
    if clips.is_empty() {
        return Err((
            1,
            "no weapon clips in the input (see --weapon-clips)".into(),
        ));
    }
    let verts = motion_core::contact::skinned_rest(&doc, &rig).map_err(|e| (3, e))?;
    let height = verts.iter().map(|v| v.p.y).fold(f64::MIN, f64::max)
        - verts.iter().map(|v| v.p.y).fold(f64::MAX, f64::min);
    let ranked = grip_search(
        &doc,
        &rig,
        hand,
        origin,
        opts.weapon_length * height,
        &clips,
    )
    .map_err(|e| (3, e))?;
    let frames: usize = clips.iter().map(|c| c.len()).sum();
    let mut r = format!(
        "=== motionforge grip ===\nclips: {}\nframes: {}\n",
        names.join(", "),
        frames
    );
    let cur = rig.rest_world[socket]
        .apply_linear(motion_core::math::Vec3::new(0.0, 1.0, 0.0))
        .normalized();
    let cur_score = grip_search_one(&ranked, cur);
    r.push_str(&format!(
        "current socket blade: ({:.3}, {:.3}, {:.3}) nearest candidate hits {}\n",
        cur.x, cur.y, cur.z, cur_score
    ));
    for (d, f, clear) in ranked.iter().take(6) {
        r.push_str(&format!(
            "candidate ({:.3}, {:.3}, {:.3}): blade hits {}, mean clearance {:.3}\n",
            d.x, d.y, d.z, f, clear
        ));
    }
    let worst = ranked.last().unwrap();
    r.push_str(&format!(
        "worst ({:.3}, {:.3}, {:.3}): hits {}\n",
        worst.0.x, worst.0.y, worst.0.z, worst.1
    ));
    if let Some(out) = cmd.get("output") {
        let best = ranked[0].0;
        let mut doc = doc.clone();
        let hw = rig.rest_world[hand];
        let hdir = (hw.apply_linear(motion_core::math::Vec3::new(0.0, 1.0, 0.0))).normalized();
        let mut x = best.cross(hdir);
        if x.length() < 1e-6 {
            x = motion_core::math::Vec3::new(1.0, 0.0, 0.0);
        }
        let x = x.normalized();
        let m = motion_core::math::Mat3::from_cols(x, best, x.cross(best));
        let local = motion_core::glb::canonical_quat(
            hw.rotation()
                .conj()
                .mul(motion_core::math::Quat::from_mat3(m)),
        );
        let node = &mut doc.array_mut("nodes")[socket];
        let (t, _, sc) = motion_core::glb::node_trs(node);
        motion_core::glb::set_node_trs(node, t, local, sc);
        save(&doc, out)?;
        r.push_str(&format!("output: {}\n", out));
    }
    Ok(r)
}

fn grip_search_one(
    ranked: &[(motion_core::math::Vec3, usize, f64)],
    d: motion_core::math::Vec3,
) -> usize {
    ranked
        .iter()
        .max_by(|a, b| {
            a.0.dot(d)
                .partial_cmp(&b.0.dot(d))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|x| x.1)
        .unwrap_or(0)
}

pub fn cmd_pose_test(cmd: &Cmd) -> CmdResult {
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let doc = load(input)?;
    let samples = pose_samples(&doc, cmd)?;
    let (posed, report) = posetest::build(&doc, &samples).map_err(|e| (3, e))?;
    save(&posed, output)?;
    let mut r = String::from("=== motionforge pose-test ===\n");
    r.push_str(&format!(
        "poses: {}\n",
        report
            .get("poses")
            .and_then(Json::as_arr)
            .map(|a| a.len())
            .unwrap_or(0)
    ));
    if let Some(sheet) = cmd.get("sheet") {
        render_sheet(cmd, output, sheet, &report)?;
        r.push_str(&format!("sheet: {}\n", sheet));
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

fn pose_samples(
    doc: &Document,
    cmd: &Cmd,
) -> Result<Vec<(String, Vec<Option<motion_core::rig::Trs>>)>, (i32, String)> {
    let sources = clip_sources(cmd)?;
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let (animated, _) = retarget_clips(doc, &sources, 30.0).map_err(|e| (3, e))?;
    posetest::clip_samples(&animated, 3, &[0.25, 0.6]).map_err(|e| (3, e))
}

pub fn cmd_fixture_glb(cmd: &Cmd) -> CmdResult {
    let output = cmd.required("output")?;
    let naming = Naming::parse(cmd.get("naming").unwrap_or("mixamo"))
        .ok_or((1, "--naming must be mixamo|plain|def".to_string()))?;
    let twisted = cmd.flag("twisted");
    let walk = cmd.flag("walk");
    let doc = fixture::humanoid(naming, twisted, walk).map_err(|e| (3, e))?;
    save(&doc, output)?;
    Ok(format!(
        "=== motionforge fixture-glb ===\noutput: {}\n",
        output
    ))
}

fn find_blender(cmd: &Cmd) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(b) = cmd.get("blender") {
        candidates.push(b.into());
    }
    for var in ["BLENDER_BIN", "BLENDER"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                candidates.push(v.into());
            }
        }
    }
    candidates.push("/Applications/Blender.app/Contents/MacOS/Blender".into());
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            candidates.push(dir.join("blender"));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn render_sheet(
    cmd: &Cmd,
    posed_glb: &str,
    sheet: &str,
    report: &Json,
) -> Result<(), (i32, String)> {
    let blender = find_blender(cmd).ok_or((
        2,
        "Blender not found (set --blender or BLENDER_BIN)".to_string(),
    ))?;
    // Blender resolves relative paths against the blend file, not the cwd.
    let abs = |p: &str| {
        std::path::absolute(p)
            .map(|a| a.to_string_lossy().to_string())
            .unwrap_or_else(|_| p.to_string())
    };
    let (posed_glb, sheet) = (abs(posed_glb), abs(sheet));
    let (posed_glb, sheet) = (posed_glb.as_str(), sheet.as_str());
    let script =
        std::env::temp_dir().join(format!("motionforge-pose-sheet-{}.py", std::process::id()));
    std::fs::write(&script, POSE_SHEET_PY)
        .map_err(|e| (3, format!("cannot write {}: {}", script.display(), e)))?;
    let fwd: Vec<String> = report
        .get("forward")
        .and_then(Json::as_arr)
        .unwrap_or(&[])
        .iter()
        .filter_map(Json::as_f64)
        .map(|v| format!("{}", round(v, 6)))
        .collect();
    let ran = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(posed_glb)
        .arg(sheet)
        .args(["--forward", &fwd.join(",")])
        .output();
    let _ = std::fs::remove_file(&script);
    let ran = ran.map_err(|e| (2, format!("cannot run Blender: {}", e)))?;
    let stdout = String::from_utf8_lossy(&ran.stdout);
    if !ran.status.success()
        || !stdout.contains("MOTIONFORGE_SHEET_OK")
        || !Path::new(sheet).is_file()
    {
        let err = String::from_utf8_lossy(&ran.stderr);
        let tail: String = err
            .lines()
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err((2, format!("pose sheet render failed:\n{}", tail)));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// genforge adapter

const ADAPTER_HELP: &str = "usage: motionforge adapter <standardize|animate|pose-test> IN.glb OUT.glb RESULT.json [flags]\n\
\n\
  standardize  rename/reparent the rig onto the standard humanoid skeleton (DEF-* names)\n\
  animate      retarget clip GLBs onto the character (--clips required), then\n\
               the contact pass (hands, forearms, held weapon out of the body)\n\
  pose-test    range-of-motion sheet (+ clip samples with --clips) for owner approval\n\
\n\
flags: --class humanoid|quadruped|custom (default humanoid)\n\
       --clips <glb|folder> (repeatable)  --fps <n> (animate, default 30)\n\
       --blender <path> (pose-test; default BLENDER_BIN, the macOS app, PATH)\n\
       animate: --no-pin, --no-contact, --weapon R|L|none, --weapon-length, --weapon-clips\n\
\n\
RESULT.json: {\"ok\": bool, \"outputs\": [paths relative to its folder], \"tool\": \"motionforge\", ...}\n\
exit: 0 ok, 1 stage failed (ok false, result written), 2 error (no result)\n";

fn relative(path: &Path, base: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let (p, b) = (abs(path), abs(base));
    match p.strip_prefix(&b) {
        Ok(rel) => rel.to_string_lossy().to_string(),
        Err(_) => p.to_string_lossy().to_string(),
    }
}

struct AdapterRun {
    ok: bool,
    outputs: Vec<PathBuf>,
    details: Json,
    reason: Option<String>,
}

fn write_result(stage: &str, result: &Path, run: &AdapterRun) -> Result<(), (i32, String)> {
    let base = result
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut pairs = vec![
        ("ok", Json::Bool(run.ok)),
        (
            "outputs",
            Json::Arr(
                run.outputs
                    .iter()
                    .map(|p| Json::str(&relative(p, &base)))
                    .collect(),
            ),
        ),
        ("tool", Json::str("motionforge")),
        ("version", Json::str(VERSION)),
        ("stage", Json::str(stage)),
    ];
    if let Some(r) = &run.reason {
        pairs.push(("reason", Json::str(r)));
    }
    pairs.push((stage, run.details.clone()));
    std::fs::write(result, pretty(&Json::obj(pairs))?)
        .map_err(|e| (2, format!("cannot write {}: {}", result.display(), e)))
}

fn adapter_standardize(input: &str, out: &str, cmd: &Cmd) -> Result<AdapterRun, (i32, String)> {
    let class = class_of(cmd)?;
    let doc = load(input)?;
    Ok(match standardize(&doc, &class).map_err(|e| (2, e))? {
        Outcome::Done(std_doc, report) => {
            save(&std_doc, out)?;
            AdapterRun {
                ok: true,
                outputs: vec![out.into()],
                details: report.to_json(),
                reason: None,
            }
        }
        Outcome::Refused(report, why) => AdapterRun {
            ok: false,
            outputs: vec![],
            details: report.to_json(),
            reason: Some(why),
        },
    })
}

fn humanoid_only(cmd: &Cmd, stage: &str) -> Result<Option<AdapterRun>, (i32, String)> {
    let class = class_of(cmd)?;
    if class == "humanoid" {
        return Ok(None);
    }
    Ok(Some(AdapterRun {
        ok: false,
        outputs: vec![],
        details: Json::obj(vec![("class", Json::str(&class))]),
        reason: Some(format!(
            "{} supports humanoids only (v1); {} rigs need their own clips and poses",
            stage, class
        )),
    }))
}

fn adapter_animate(input: &str, out: &str, cmd: &Cmd) -> Result<AdapterRun, (i32, String)> {
    if let Some(run) = humanoid_only(cmd, "animate")? {
        return Ok(run);
    }
    let fps = crate::parse_f64(cmd, "fps", 30.0)?;
    let sources = clip_sources(cmd)?;
    if sources.is_empty() {
        return Err((
            2,
            "animate needs --clips (a clip GLB or a folder of them)".into(),
        ));
    }
    let doc = load(input)?;
    let contact = contact_opts(cmd)?;
    let (animated, reports) = match retarget_with(
        &doc,
        &sources,
        fps,
        None,
        contact.as_ref(),
        !cmd.flag("no-pin"),
    ) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AdapterRun {
                ok: false,
                outputs: vec![],
                details: Json::obj(vec![]),
                reason: Some(e),
            })
        }
    };
    save(&animated, out)?;
    let worst = reports.iter().map(|r| r.max_error_deg).fold(0.0, f64::max);
    let ok = !reports.is_empty() && worst < 0.5;
    Ok(AdapterRun {
        ok,
        outputs: vec![out.into()],
        details: Json::obj(vec![
            (
                "clips",
                Json::Arr(reports.iter().map(|r| r.to_json()).collect()),
            ),
            (
                "sources",
                Json::Arr(sources.iter().map(|(l, _)| Json::str(l)).collect()),
            ),
            ("max_error_deg", Json::num(round(worst, 6))),
        ]),
        reason: if ok {
            None
        } else {
            Some("no clips retargeted, or the transfer self-check exceeded 0.5 deg".into())
        },
    })
}

fn adapter_pose_test(input: &str, out: &str, cmd: &Cmd) -> Result<AdapterRun, (i32, String)> {
    if let Some(run) = humanoid_only(cmd, "pose-test")? {
        return Ok(run);
    }
    let dir = Path::new(out)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let doc = load(input)?;
    let samples = pose_samples(&doc, cmd)?;
    let (posed, report) = match posetest::build(&doc, &samples) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AdapterRun {
                ok: false,
                outputs: vec![],
                details: Json::obj(vec![]),
                reason: Some(e),
            })
        }
    };
    let posed_path = dir.join("pose-test.glb");
    let sheet = dir.join("pose-sheet.png");
    save(&posed, &posed_path.to_string_lossy())?;
    render_sheet(
        cmd,
        &posed_path.to_string_lossy(),
        &sheet.to_string_lossy(),
        &report,
    )?;
    // The model itself is unchanged by a pose test: pass it on as is.
    std::fs::copy(input, out)
        .map_err(|e| (2, format!("cannot copy {} -> {}: {}", input, out, e)))?;
    let mut details = report;
    details.set("posed_glb", Json::str("pose-test.glb"));
    details.set("clip_samples", Json::num(samples.len() as f64));
    Ok(AdapterRun {
        ok: true,
        // Sheet first (genforge's preview), model last (the next stage's input).
        outputs: vec![sheet, out.into()],
        details,
        reason: None,
    })
}

/// `motionforge adapter ...`; returns the process exit code.
pub fn adapter_main(argv: &[String]) -> i32 {
    if argv.len() < 4 || argv.iter().any(|a| a == "--help" || a == "-h") {
        eprint!("{}", ADAPTER_HELP);
        return if argv.iter().any(|a| a == "--help" || a == "-h") {
            0
        } else {
            2
        };
    }
    let (stage, input, out, result) = (&argv[0], &argv[1], &argv[2], &argv[3]);
    let mut flag_argv = vec!["adapter".to_string()];
    flag_argv.extend(argv[4..].iter().cloned());
    let cmd = match parse_argv(&flag_argv) {
        Ok((_, c)) => c,
        Err((_, msg)) => {
            eprintln!("motionforge adapter: {}", msg);
            return 2;
        }
    };
    let ran = match stage.as_str() {
        "standardize" => adapter_standardize(input, out, &cmd),
        "animate" => adapter_animate(input, out, &cmd),
        "pose-test" => adapter_pose_test(input, out, &cmd),
        other => Err((2, format!("unknown adapter stage '{}'", other))),
    };
    let run = match ran {
        Ok(run) => run,
        Err((_, msg)) => {
            eprintln!("motionforge adapter {}: {}", stage, msg);
            return 2;
        }
    };
    if let Err((_, msg)) = write_result(stage, Path::new(result), &run) {
        eprintln!("motionforge adapter {}: {}", stage, msg);
        return 2;
    }
    println!(
        "motionforge adapter {}: {}{}",
        stage,
        if run.ok { "ok" } else { "failed" },
        run.reason
            .as_deref()
            .map(|r| format!(" ({})", r))
            .unwrap_or_default()
    );
    if run.ok {
        0
    } else {
        1
    }
}

/// `motion check --camera game`: readability of each scored clip at the
/// game's follow camera (`motion_core::readcam`).
fn read_at_camera(
    cmd: &Cmd,
    cam: &str,
    doc: &Document,
    subject: &motion_core::motioncheck::Subject,
    scores: &mut [motion_core::motioncheck::ClipScore],
    fps: Option<f64>,
) -> Result<(), (i32, String)> {
    use motion_core::motioncheck::sample;
    use motion_core::readcam::{BladePose, CamOptions, Stage};
    use motion_core::rig::Animation;
    if cam != "game" {
        return Err((1, format!("--camera must be game, got '{}'", cam)));
    }
    let mut o = CamOptions::default();
    o.yaw_deg = crate::parse_f64(cmd, "cam-yaw", o.yaw_deg)?;
    o.stature_m = crate::parse_f64(cmd, "stature", o.stature_m)?;
    o.blade_to_m = crate::parse_f64(cmd, "blade-length", o.blade_to_m)?;
    o.blade_width_m = crate::parse_f64(cmd, "blade-width", o.blade_width_m)?;
    if let Some(p) = cmd.get("blade-pose") {
        o.blade_pose = match p {
            "game" => BladePose::Game,
            "socket" => BladePose::Socket,
            other => {
                return Err((
                    1,
                    format!("--blade-pose must be game|socket, got '{}'", other),
                ))
            }
        };
    }
    if !(0.0..=10.0).contains(&o.stature_m) || !(0.0..=5.0).contains(&o.blade_to_m) {
        return Err((
            1,
            "--stature must be in 0..10 and --blade-length in 0..5".into(),
        ));
    }
    let stage = Stage::new(doc, subject, &o).map_err(|e| (3, e))?;
    let anims = Animation::load_all(doc).map_err(|e| (3, e))?;
    // Neutral: each clip's own first frame, or `--neutral <clip>`'s.
    let neutral = match cmd.get("neutral") {
        Some(want) => Some(
            anims
                .iter()
                .find(|a| a.name == want)
                .map(|a| sample(&subject.rig, a, fps).1.swap_remove(0))
                .ok_or((2, format!("no clip named '{}' for --neutral", want)))?,
        ),
        None => None,
    };
    let dir = cmd.get("silhouettes");
    for sc in scores.iter_mut() {
        let Some(anim) = anims.iter().find(|a| a.name == sc.name) else {
            continue;
        };
        let (times, frames) = sample(&subject.rig, anim, fps);
        let dt = if times.len() > 1 {
            (times[times.len() - 1] - times[0]) / (times.len() - 1) as f64
        } else {
            1.0 / 30.0
        };
        let base = neutral.clone().unwrap_or_else(|| frames[0].clone());
        let (rc, shots) = stage.read(&subject.rig, &frames, dt, &base);
        if let Some(d) = dir {
            std::fs::create_dir_all(d).map_err(|e| (3, e.to_string()))?;
            for (f, s) in shots.iter().enumerate() {
                let path = format!("{}/{}.{:04}.pgm", d, sc.name, f);
                std::fs::write(&path, s.body.to_pgm(s.blade.as_ref()))
                    .map_err(|e| (3, e.to_string()))?;
            }
        }
        sc.read_cam = Some(rc);
    }
    Ok(())
}

/// `motionforge motion check <glb>`: score every clip of a rigged GLB
/// (`motion_core::motioncheck`; `docs/motion-check.md`).
pub fn cmd_motion_check(cmd: &Cmd) -> CmdResult {
    use motion_core::motioncheck::{check_doc, ATTACK_WORDS};
    let input = cmd.required("input")?;
    let doc = load(input)?;
    let opts = contact_opts(cmd)?.unwrap_or_default();
    let only: Vec<String> = cmd
        .all("clip")
        .iter()
        .flat_map(|c| c.split(','))
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    let words: Vec<String> = match cmd.get("attack-clips") {
        Some(list) => list
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect(),
        None => ATTACK_WORDS.iter().map(|w| w.to_string()).collect(),
    };
    let fps = match cmd.get("fps") {
        Some(_) => {
            let f = crate::parse_f64(cmd, "fps", 30.0)?;
            if !(1.0..=240.0).contains(&f) {
                return Err((1, "--fps must be in 1..240".into()));
            }
            Some(f)
        }
        None => None,
    };
    let (subject, mut scores) = check_doc(&doc, &opts, &only, &words, fps).map_err(|e| (3, e))?;
    if let Some(cam) = cmd.get("camera") {
        read_at_camera(cmd, cam, &doc, &subject, &mut scores, fps)?;
    }
    for o in &only {
        if !scores.iter().any(|s| &s.name == o) {
            return Err((2, format!("no clip named '{}'", o)));
        }
    }
    if cmd.flag("json") {
        let j = Json::obj(vec![
            ("height_m", Json::num(round(subject.height, 4))),
            ("scale_k", Json::num(round(subject.k, 4))),
            (
                "clips",
                Json::Arr(scores.iter().map(|s| s.to_json()).collect()),
            ),
        ]);
        return pretty(&j);
    }
    let mut r = String::from("=== motionforge motion check ===\n");
    r.push_str(&format!(
        "character {:.3} m tall; lengths as on a 1.75 m body; jerk m/s^3 mean/peak (worst chain), root mean\n",
        subject.height
    ));
    for s in &scores {
        r.push_str(&s.line());
    }
    if scores.len() > 1 {
        let n = |s: &motion_core::motioncheck::ClipScore| s.frames.max(1) as f64;
        let mut by: Vec<(&str, Vec<(String, f64)>)> = vec![
            (
                "skate (share of frames)",
                scores
                    .iter()
                    .map(|s| (s.name.clone(), s.feet.skate_frames as f64 / n(s)))
                    .collect(),
            ),
            (
                "through the body (share of frames)",
                scores
                    .iter()
                    .map(|s| {
                        (
                            s.name.clone(),
                            s.through
                                .as_ref()
                                .map(|t| t.mesh_frames as f64 / n(s))
                                .unwrap_or(0.0),
                        )
                    })
                    .collect(),
            ),
            (
                "jerk (worst chain mean)",
                scores
                    .iter()
                    .map(|s| (s.name.clone(), s.jerk_worst().1))
                    .collect(),
            ),
        ];
        r.push_str("worst:\n");
        for (what, list) in by.iter_mut() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            let top: Vec<String> = list
                .iter()
                .take(3)
                .filter(|(_, v)| *v > 0.0)
                .map(|(n, v)| {
                    if what.starts_with("jerk") {
                        format!("{} {:.0}", n, v)
                    } else {
                        format!("{} {:.0}%", n, v * 100.0)
                    }
                })
                .collect();
            r.push_str(&format!(
                "  {}: {}\n",
                what,
                if top.is_empty() {
                    "none".into()
                } else {
                    top.join(", ")
                }
            ));
        }
    }
    Ok(r)
}

/// `motionforge socket`: put the hand sockets in the palms (hands with
/// finger bones), where mocap sword clips hold the handle.
pub fn cmd_socket(cmd: &Cmd) -> CmdResult {
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let mut doc = load(input)?;
    let placed = motion_core::attack::grip::place_sockets(&mut doc).map_err(|e| (3, e))?;
    if placed.is_empty() {
        return Err((3, "no hand has finger bones: nothing placed".into()));
    }
    save(&doc, output)?;
    Ok(format!(
        "=== motionforge socket ===\nplaced in the palm: {}\noutput: {}\n",
        placed.join(", "),
        output
    ))
}

/// `motionforge attack`: a sword attack authored from the sword's path
/// (`motion_core::attack`), solved onto a rigged GLB as a clip.
pub fn cmd_attack(cmd: &Cmd) -> CmdResult {
    use motion_core::attack::{build, parse_spec, preset, PRESETS};
    let input = cmd.required("rig")?;
    let output = cmd.required("output")?;
    let mut doc = load(input)?;
    let mut specs: Vec<String> = Vec::new();
    for p in cmd.all("preset").iter().flat_map(|p| p.split(',')) {
        let p = p.trim();
        if p == "all" {
            specs.extend(PRESETS.iter().map(|n| preset(n).unwrap().to_string()));
        } else if !p.is_empty() {
            specs.push(
                preset(p)
                    .ok_or((
                        1,
                        format!("no preset '{}' (have: {})", p, PRESETS.join(", ")),
                    ))?
                    .to_string(),
            );
        }
    }
    for path in cmd.all("spec") {
        specs.push(std::fs::read_to_string(path).map_err(|e| (2, format!("{path}: {e}")))?);
    }
    if specs.is_empty() {
        return Err((1, "give --spec <json> or --preset <name|all>".into()));
    }
    let prefix = cmd.get("prefix").unwrap_or("");
    let mut r = String::from("=== motionforge attack ===\n");
    let mut reports = Vec::new();
    let fist = match cmd.get("grips") {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| (2, format!("{path}: {e}")))?;
            motion_core::attack::grip::parse_grips(&text, "fist").map_err(|e| (2, e))?
        }
        None => Vec::new(),
    };
    for text in &specs {
        let mut spec = parse_spec(text).map_err(|e| (2, e))?;
        spec.fist = fist.clone();
        if let Some(c) = cmd.get("close") {
            spec.close = c
                .parse::<f64>()
                .ok()
                .filter(|v| (0.0..=1.5).contains(v))
                .ok_or((1, "--close must be 0..1.5".to_string()))?;
        }
        let name = match (cmd.get("name"), specs.len()) {
            (Some(n), 1) => n.to_string(),
            _ => format!("{}{}", prefix, spec.name),
        };
        let (out, rep) = build(&doc, &spec, Some(&name)).map_err(|e| (3, e))?;
        doc = out;
        r.push_str(&format!(
            "clip {}: {} frames at {} fps, keys {}; grip off by up to {:.1}% of height (frame {}), blade off by up to {:.1} deg, hips lowered {:.1}% so the legs reach, feet off by {:.2}%\n",
            rep.name,
            rep.frames,
            rep.fps,
            rep.keys
                .iter()
                .zip(&rep.key_miss)
                .map(|((f, role), m)| format!(
                    "{role}@{f}[grip {:.1}% blade {:.0}deg]",
                    m.grip * 100.0,
                    m.blade_deg
                ))
                .collect::<Vec<_>>()
                .join(" "),
            rep.worst.grip * 100.0,
            rep.worst_grip_frame,
            rep.worst.blade_deg,
            rep.worst.drop * 100.0,
            rep.worst.foot * 100.0
        ));
        reports.push(rep.to_json());
    }
    save(&doc, output)?;
    if let Some(path) = cmd.get("report") {
        std::fs::write(path, pretty(&Json::Arr(reports))?).map_err(|e| (3, e.to_string()))?;
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// `motionforge clean`: the clean-up pass on a GLB's own clips
/// (`motion_core::clean`).
pub fn cmd_clean(cmd: &Cmd) -> CmdResult {
    use motion_core::clean::{clean_glb, CleanOptions};
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let opts = CleanOptions {
        ground: !cmd.flag("no-ground"),
        lock: !cmd.flag("no-lock"),
        contact: if cmd.flag("contact") {
            contact_opts(cmd)?.map(|mut o| {
                if cmd.get("contact-ramp").is_none() {
                    o.ramp = 0.3;
                }
                o.min_depth = 0.02;
                o
            })
        } else {
            None
        },
    };
    let clips: Vec<String> = cmd
        .all("clip")
        .iter()
        .flat_map(|c| c.split(','))
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    let doc = load(input)?;
    let (out, reports) = clean_glb(&doc, &opts, &clips, cmd.get("suffix")).map_err(|e| (3, e))?;
    save(&out, output)?;
    if let Some(path) = cmd.get("report") {
        let j = Json::Arr(reports.iter().map(|r| r.to_json()).collect());
        std::fs::write(path, pretty(&j)?).map_err(|e| (3, e.to_string()))?;
    }
    let mut r = String::from("=== motionforge clean ===\n");
    for c in &reports {
        r.push_str(&format!(
            "clip {}: {} frames, ground {:+.1} cm, hips up to {:.1} cm, lifted {} foot-frames (up to {:.1} cm), {} plants locked over {} frames (ankle moved up to {:.1} cm)\n",
            c.name,
            c.frames,
            c.ground_shift_m * 100.0,
            c.hips_lift_max_m * 100.0,
            c.lift_frames,
            c.lift_max_m * 100.0,
            c.plants_locked,
            c.lock_frames,
            c.lock_max_m * 100.0
        ));
        if let Some(ct) = &c.contact {
            r.push_str(&contact_line(ct));
        }
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// `motionforge stylize --input <rigged glb>`: the stylizer on a GLB's
/// own clips (`motion_core::stylize_body`).
pub fn cmd_stylize_glb(cmd: &Cmd, base: motion_core::stylize::StylizeParams) -> CmdResult {
    use motion_core::stylize_body::{stylize_glb, BodyParams, Mode};
    let input = cmd.required("input")?;
    let output = cmd.required("output")?;
    let mode = match cmd.get("mode").unwrap_or("body") {
        "body" => Mode::Body,
        "per-bone" => Mode::PerBone,
        other => return Err((1, format!("--mode must be body|per-bone, got '{}'", other))),
    };
    let params = BodyParams {
        base,
        key_prominence: crate::parse_f64(cmd, "key-prominence", 0.25)?,
        snap_speed: crate::parse_f64(cmd, "snap-speed", 6.0)?,
        accent_side: match cmd.get("accent").unwrap_or("R") {
            "R" | "r" => Some("R".into()),
            "L" | "l" => Some("L".into()),
            "none" => None,
            other => return Err((1, format!("--accent must be R|L|none, got '{}'", other))),
        },
        contact: if cmd.flag("contact") {
            contact_opts(cmd)?
        } else {
            None
        },
        ..BodyParams::default()
    };
    let clips: Vec<String> = cmd
        .all("clip")
        .iter()
        .flat_map(|c| c.split(','))
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    let doc = load(input)?;
    let replant = !cmd.flag("no-replant") && mode == Mode::Body;
    let (out, reports) =
        stylize_glb(&doc, &params, mode, &clips, cmd.get("suffix"), replant).map_err(|e| (3, e))?;
    save(&out, output)?;
    let mut r = format!(
        "=== motionforge stylize ({}) ===\n",
        if mode == Mode::Body {
            "body"
        } else {
            "per-bone"
        }
    );
    for c in &reports {
        r.push_str(&format!(
            "clip {} (from {}): {} frames, keys {:?}, big {:?}, anticipation {} added {} skipped",
            c.name, c.source, c.frames, c.keys, c.big, c.counters_added, c.counters_skipped
        ));
        if c.exag_scale < 1.0 {
            r.push_str(&format!(
                ", exaggeration x{:.2} (a fast snap)",
                c.exag_scale
            ));
        }
        if replant {
            r.push_str(&format!(
                ", feet re-planted: {} plants, {} frames, ankle moved up to {:.1} cm",
                c.plants,
                c.replant_frames,
                c.replant_max_m * 100.0
            ));
        }
        r.push('\n');
        if let Some(ct) = &c.contact {
            r.push_str(&contact_line(ct));
        }
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}
