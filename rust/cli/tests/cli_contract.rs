// CLI contract test: stdout reports (output paths normalized to <OUT>)
// and output files are pinned against goldens in tests/fixtures/golden/.
// Regenerate deliberately after an intended change:
//   UPDATE_GOLDENS=1 cargo test --locked --release -p motionforge --test cli_contract
// then review the diff.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
}

fn golden_dir() -> PathBuf {
    fixtures_dir().join("golden")
}

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_motionforge"))
}

fn workdir(case: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("motionforge-cli-{}_{}", std::process::id(), case));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[String]) -> (i32, String, String) {
    let out = Command::new(binary())
        .args(args)
        .output()
        .expect("spawn cli");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn check(case: &str, args: &[String], outputs: &[&str]) {
    let update = std::env::var("UPDATE_GOLDENS").is_ok();
    let (code, stdout, stderr) = run(args);
    assert_eq!(code, 0, "case {} failed: stderr:\n{}", case, stderr);
    assert!(
        stderr.is_empty(),
        "case {} wrote to stderr:\n{}",
        case,
        stderr
    );
    // Normalize every output path in stdout to <OUT> for machine
    // independence (the files themselves are compared byte-for-byte).
    let mut norm = stdout;
    for path in outputs {
        norm = norm.replace(path, "<OUT>");
    }
    let stdout_golden = golden_dir().join(format!("{}.stdout", case));
    if update {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&stdout_golden, &norm).unwrap();
    } else {
        let want = std::fs::read_to_string(&stdout_golden)
            .unwrap_or_else(|_| panic!("missing golden for {}", case));
        assert_eq!(norm, want, "case {} stdout drifted", case);
    }
    for (i, path) in outputs.iter().enumerate() {
        let bytes = std::fs::read(path).unwrap();
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("out");
        let golden = golden_dir().join(format!("{}.{}.{}", case, i, ext));
        if update {
            std::fs::write(&golden, &bytes).unwrap();
        } else {
            let want = std::fs::read(&golden)
                .unwrap_or_else(|_| panic!("missing golden for {}:{}", case, i));
            assert_eq!(bytes, want, "case {} output {} drifted", case, i);
        }
    }
}

fn fixture(name: &str) -> String {
    fixtures_dir().join(name).to_str().unwrap().to_string()
}

#[test]
fn contract_clip_info() {
    let dir = workdir("clip_info");
    let skel = dir.join("skel.json").to_str().unwrap().to_string();
    check(
        "clip_info",
        &[
            "clip-info".to_string(),
            "--input".to_string(),
            fixture("walk_src.json"),
            "--emit-skeleton".to_string(),
            skel.clone(),
        ],
        &[&skel],
    );
}

#[test]
fn contract_retarget() {
    let dir = workdir("retarget");
    let out = dir.join("hero.json").to_str().unwrap().to_string();
    check(
        "retarget",
        &[
            "retarget".to_string(),
            "--source".to_string(),
            fixture("walk_src.json"),
            "--target".to_string(),
            fixture("hero_skel.json"),
            "--map".to_string(),
            fixture("hero_map.json"),
            "--output".to_string(),
            out.clone(),
        ],
        &[&out],
    );
}

#[test]
fn contract_stylize() {
    let dir = workdir("stylize");
    let out = dir.join("styl.json").to_str().unwrap().to_string();
    let keys = dir.join("keys.json").to_str().unwrap().to_string();
    check(
        "stylize",
        &[
            "stylize".to_string(),
            "--input".to_string(),
            fixture("walk_src.json"),
            "--output".to_string(),
            out.clone(),
            "--chain".to_string(),
            "Arm:1.8".to_string(),
            "--keys-out".to_string(),
            keys.clone(),
        ],
        &[&out, &keys],
    );
}

#[test]
fn contract_physics_check() {
    check(
        "physics_check",
        &[
            "physics-check".to_string(),
            "--input".to_string(),
            fixture("jump.json"),
            "--root".to_string(),
            "DEF-spine".to_string(),
            "--feet".to_string(),
            "DEF-foot.L,DEF-foot.R".to_string(),
        ],
        &[],
    );
}

#[test]
fn contract_physics_frame() {
    check(
        "physics_frame",
        &[
            "physics-frame".to_string(),
            "--input".to_string(),
            fixture("jump.json"),
            "--frame".to_string(),
            "1".to_string(),
            "--root".to_string(),
            "DEF-spine".to_string(),
            "--feet".to_string(),
            "DEF-foot.L,DEF-foot.R".to_string(),
        ],
        &[],
    );
}

#[test]
fn contract_physics_fix() {
    let dir = workdir("physics_fix");
    let out = dir.join("fixed.json").to_str().unwrap().to_string();
    check(
        "physics_fix",
        &[
            "physics-fix".to_string(),
            "--input".to_string(),
            fixture("jump.json"),
            "--root".to_string(),
            "DEF-spine".to_string(),
            "--feet".to_string(),
            "DEF-foot.L,DEF-foot.R".to_string(),
            "--output".to_string(),
            out.clone(),
        ],
        &[&out],
    );
}

#[test]
fn contract_autopose() {
    let dir = workdir("autopose");
    let out = dir.join("pose.json").to_str().unwrap().to_string();
    check(
        "autopose",
        &[
            "autopose".to_string(),
            "--model".to_string(),
            fixture("autopose_weights.json"),
            "--effectors".to_string(),
            fixture("autopose_effectors.json"),
            "--output".to_string(),
            out.clone(),
        ],
        &[&out],
    );
}

#[test]
fn contract_autopose_limits() {
    let dir = workdir("autopose_limits");
    let out = dir.join("pose.json").to_str().unwrap().to_string();
    check(
        "autopose_limits",
        &[
            "autopose".to_string(),
            "--model".to_string(),
            fixture("autopose_weights.json"),
            "--effectors".to_string(),
            fixture("autopose_effectors.json"),
            "--limits".to_string(),
            fixture("limits_hero.json"),
            "--output".to_string(),
            out.clone(),
        ],
        &[&out],
    );
}

#[test]
fn contract_help_and_errors() {
    let (code, stdout, stderr) = run(&["--help".to_string()]);
    assert!(stderr.is_empty());
    assert_eq!(code, 0);
    assert!(stdout.contains("motionforge") && stdout.contains("retarget"));
    let (code, _, stderr) = run(&["nope".to_string()]);
    assert_eq!(code, 1);
    assert!(stderr.contains("unknown command"));
    // Unreadable input beats missing flags (input error, exit 2).
    let (code, _, stderr) = run(&[
        "retarget".to_string(),
        "--source".to_string(),
        "x".to_string(),
    ]);
    assert_eq!(code, 2);
    assert!(stderr.contains("cannot read"));
    let (code, _, stderr) = run(&[
        "retarget".to_string(),
        "--source".to_string(),
        fixture("walk_src.json"),
    ]);
    assert_eq!(code, 1);
    assert!(stderr.contains("missing required"));
    let (code, _, stderr) = run(&[
        "clip-info".to_string(),
        "--input".to_string(),
        "/nonexistent.json".to_string(),
    ]);
    assert_eq!(code, 2);
    assert!(stderr.contains("cannot read"));
}
