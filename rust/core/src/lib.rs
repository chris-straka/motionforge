//! motionforge core: deterministic animation math (MIT).
//!
//! Clip I/O, retarget transfer, stylizer, physics pass, AutoPose
//! inference, and GLB rig adapters (standardize, animate, pose test). Zero dependencies; byte-deterministic outputs.

pub mod animate;
pub mod autopose;
pub mod clip;
pub mod detmath;
pub mod fixture;
pub mod glb;
pub mod helpers;
pub mod humanoid;
pub mod json;
pub mod limits;
pub mod math;
pub mod physics;
pub mod posetest;
pub mod retarget;
pub mod rig;
pub mod standardize;
pub mod stylize;
