//! Creature motion: AI video of a creature in, an animation on its own
//! skeleton out, for bodies with no mocap. The joint finder (a per-
//! creature DINOv2 fine-tune) lives in python/creature/; this module is
//! everything deterministic around it: the creature profile from a rig,
//! the two known cameras, random poses for training, the exam motion,
//! triangulation, the solve onto the skeleton (limits, smoothing,
//! planted feet), scoring against the truth, and the GLB animation.
//! Spec: docs/creature.md.

pub mod camera;
pub mod compare;
pub mod detect;
pub mod glbanim;
pub mod motion;
pub mod profile;
pub mod rigcheck;
pub mod rng;
pub mod solve;
