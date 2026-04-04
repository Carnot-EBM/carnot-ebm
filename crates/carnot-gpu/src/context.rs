//! GPU device context: initialization and resource management.
//!
//! Wraps wgpu device/queue setup. Creates one per application, shared across
//! all GPU-accelerated operations.

use thiserror::Error;

/// Errors from GPU operations.
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No compatible GPU adapter found")]
    NoAdapter,
    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),
    #[error("GPU buffer mapping failed")]
    BufferMap,
}

/// GPU device context: holds wgpu instance, adapter, device, and queue.
///
/// Create once, pass to GPU-accelerated operations like `IsingGpu`.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize GPU context. Prefers high-performance discrete GPUs,
    /// falls back to integrated.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        tracing::info!(
            "GPU adapter: {} ({:?}, {:?})",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("carnot-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    /// Human-readable GPU device name (e.g., "AMD Radeon 890M Graphics").
    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// GPU backend in use (Vulkan, Metal, DX12).
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }
}
