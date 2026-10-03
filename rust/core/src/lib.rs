//! motionforge core: deterministic animation math (MIT).
//!
//! Clip I/O, retarget transfer, stylizer, physics pass, and AutoPose
//! inference. Zero dependencies; byte-deterministic outputs.

pub mod autopose;
pub mod clip;
pub mod detmath;
pub mod json;
pub mod limits;
pub mod math;
pub mod physics;
pub mod retarget;
pub mod stylize;
