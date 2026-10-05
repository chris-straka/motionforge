// genforge adapter contract: `motionforge adapter <stage> IN OUT RESULT`.
// Exit 0 ok / 1 stage failed (result with ok false) / 2 error (no result);
// RESULT.json outputs are paths relative to its folder.

use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::Document;
use motion_core::json::{parse, Json};
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_motionforge"))
}

fn workdir(case: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "motionforge-adapter-{}_{}",
        std::process::id(),
        case
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(doc: &Document, path: &Path) {
    std::fs::write(path, doc.to_bytes().unwrap()).unwrap();
}

fn run(args: &[&str]) -> i32 {
    let out = Command::new(bin()).args(args).output().unwrap();
    out.status.code().unwrap_or(-1)
}

fn result(path: &Path) -> Json {
    parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn outputs(r: &Json) -> Vec<String> {
    r.get("outputs")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn standardize_then_animate_follow_the_contract() {
    let dir = workdir("chain");
    write(
        &humanoid(Naming::Mixamo, true, false).unwrap(),
        &dir.join("rig.glb"),
    );
    write(
        &humanoid(Naming::Plain, false, true).unwrap(),
        &dir.join("walk.glb"),
    );
    let std_dir = dir.join("08-standardize");
    std::fs::create_dir_all(&std_dir).unwrap();
    let code = run(&[
        "adapter",
        "standardize",
        s(&dir.join("rig.glb")),
        s(&std_dir.join("output.glb")),
        s(&std_dir.join("result.json")),
        "--class",
        "humanoid",
    ]);
    assert_eq!(code, 0);
    let r = result(&std_dir.join("result.json"));
    assert_eq!(r.get("ok").unwrap().as_bool(), Some(true));
    assert_eq!(r.get("tool").unwrap().as_str(), Some("motionforge"));
    assert_eq!(outputs(&r), vec!["output.glb"]);
    assert!(r.get("standardize").unwrap().get("renamed").is_some());

    let anim_dir = dir.join("11-animate");
    std::fs::create_dir_all(&anim_dir).unwrap();
    let code = run(&[
        "adapter",
        "animate",
        s(&std_dir.join("output.glb")),
        s(&anim_dir.join("output.glb")),
        s(&anim_dir.join("result.json")),
        "--clips",
        s(&dir.join("walk.glb")),
    ]);
    assert_eq!(code, 0);
    let r = result(&anim_dir.join("result.json"));
    assert_eq!(outputs(&r), vec!["output.glb"]);
    let clips = r
        .get("animate")
        .unwrap()
        .get("clips")
        .unwrap()
        .as_arr()
        .unwrap();
    assert_eq!(clips[0].get("name").unwrap().as_str(), Some("walk"));
    let animated = Document::read(s(&anim_dir.join("output.glb"))).unwrap();
    assert_eq!(animated.array("animations").len(), 1);

    // A folder of clips works too.
    let lib = dir.join("clips");
    std::fs::create_dir_all(&lib).unwrap();
    std::fs::copy(dir.join("walk.glb"), lib.join("a.glb")).unwrap();
    std::fs::copy(dir.join("walk.glb"), lib.join("b.glb")).unwrap();
    let code = run(&[
        "adapter",
        "animate",
        s(&std_dir.join("output.glb")),
        s(&dir.join("two.glb")),
        s(&dir.join("two.json")),
        "--clips",
        s(&lib),
    ]);
    assert_eq!(code, 0);
    let names: Vec<String> = Document::read(s(&dir.join("two.glb")))
        .unwrap()
        .array("animations")
        .iter()
        .map(|a| a.get("name").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["walk", "walk-2"]);
}

#[test]
fn failures_report_ok_false_and_errors_write_nothing() {
    let dir = workdir("fail");
    // A model with no skin: standardize refuses (exit 1, ok false).
    let unrigged = Document {
        json: parse(r#"{"asset":{"version":"2.0"},"nodes":[{"name":"mesh"}]}"#).unwrap(),
        bin: vec![],
    };
    write(&unrigged, &dir.join("unrigged.glb"));
    let code = run(&[
        "adapter",
        "standardize",
        s(&dir.join("unrigged.glb")),
        s(&dir.join("o.glb")),
        s(&dir.join("r.json")),
    ]);
    assert_eq!(code, 1);
    let r = result(&dir.join("r.json"));
    assert_eq!(r.get("ok").unwrap().as_bool(), Some(false));
    assert!(outputs(&r).is_empty());
    assert!(r
        .get("reason")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("no skin"));
    assert!(!dir.join("o.glb").exists());

    // Unreadable input: exit 2 and no result file.
    std::fs::write(dir.join("junk.glb"), b"not a glb").unwrap();
    let code = run(&[
        "adapter",
        "standardize",
        s(&dir.join("junk.glb")),
        s(&dir.join("o2.glb")),
        s(&dir.join("r2.json")),
    ]);
    assert_eq!(code, 2);
    assert!(!dir.join("r2.json").exists());

    // animate without clips is a configuration error.
    write(
        &humanoid(Naming::Def, false, false).unwrap(),
        &dir.join("def.glb"),
    );
    let code = run(&[
        "adapter",
        "animate",
        s(&dir.join("def.glb")),
        s(&dir.join("o3.glb")),
        s(&dir.join("r3.json")),
    ]);
    assert_eq!(code, 2);
    assert!(!dir.join("r3.json").exists());

    // Non-humanoid pose tests are not supported yet: ok false, exit 1.
    let code = run(&[
        "adapter",
        "pose-test",
        s(&dir.join("def.glb")),
        s(&dir.join("o4.glb")),
        s(&dir.join("r4.json")),
        "--class",
        "quadruped",
    ]);
    assert_eq!(code, 1);
    assert_eq!(
        result(&dir.join("r4.json")).get("ok").unwrap().as_bool(),
        Some(false)
    );

    // Unknown stage / missing args.
    assert_eq!(run(&["adapter", "explode", "a", "b", "c"]), 2);
    assert_eq!(run(&["adapter", "standardize"]), 2);
}

fn blender() -> Option<PathBuf> {
    for v in ["BLENDER_BIN", "BLENDER"] {
        if let Ok(p) = std::env::var(v) {
            if Path::new(&p).is_file() {
                return Some(p.into());
            }
        }
    }
    let mac = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    mac.is_file().then_some(mac)
}

/// Renders with real Blender (skipped where Blender is absent, e.g. the
/// Linux Rust CI job).
#[test]
fn pose_test_renders_a_sheet_and_passes_the_model_on() {
    let Some(blender) = blender() else {
        eprintln!("skip: Blender not found");
        return;
    };
    let dir = workdir("pose");
    write(
        &humanoid(Naming::Def, false, false).unwrap(),
        &dir.join("def.glb"),
    );
    write(
        &humanoid(Naming::Mixamo, false, true).unwrap(),
        &dir.join("walk.glb"),
    );
    let step = dir.join("10-pose-test");
    std::fs::create_dir_all(&step).unwrap();
    let code = run(&[
        "adapter",
        "pose-test",
        s(&dir.join("def.glb")),
        s(&step.join("output.glb")),
        s(&step.join("result.json")),
        "--clips",
        s(&dir.join("walk.glb")),
        "--blender",
        s(&blender),
    ]);
    assert_eq!(code, 0);
    let r = result(&step.join("result.json"));
    assert_eq!(outputs(&r), vec!["pose-sheet.png", "output.glb"]);
    let png = std::fs::read(step.join("pose-sheet.png")).unwrap();
    assert_eq!(&png[1..4], b"PNG");
    assert!(
        png.len() > 20_000,
        "sheet looks empty ({} bytes)",
        png.len()
    );
    assert_eq!(
        std::fs::read(step.join("output.glb")).unwrap(),
        std::fs::read(dir.join("def.glb")).unwrap()
    );
    let poses = r
        .get("pose-test")
        .unwrap()
        .get("poses")
        .unwrap()
        .as_arr()
        .unwrap()
        .len();
    assert_eq!(poses, 14);
}
