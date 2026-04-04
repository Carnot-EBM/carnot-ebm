//! # carnot-gpu: GPU-accelerated compute for Carnot EBMs
//!
//! Uses wgpu (WebGPU standard) to run energy computations on any GPU:
//! AMD (Vulkan), NVIDIA (Vulkan/CUDA), Intel (Vulkan), Apple (Metal).
//!
//! The GPU is most valuable for:
//! - **Batch energy evaluation**: score thousands of configurations in parallel
//! - **Gradient computation**: autodiff via finite differences on GPU
//! - **Multi-start repair**: run N repair trajectories simultaneously
//!
//! ## Quick Start
//!
//! ```no_run
//! use carnot_gpu::GpuContext;
//!
//! let ctx = GpuContext::new().expect("No GPU available");
//! println!("GPU: {}", ctx.device_name());
//! ```

mod context;
mod ising_gpu;

pub use context::GpuContext;
pub use ising_gpu::IsingGpu;
