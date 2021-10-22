//! Tools for management of device resources and handles
use anyhow::Result;

/// Type for the information needed to interact with a given device
pub enum Device {
    Gpu {
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    },
    Cpu,
}

impl Device {
    /// Creates a new gpu device from an instance
    async fn new_gpu(instance: &wgpu::Instance,) -> Result<Self> {
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ).await?;

        Ok(Device::Gpu {
            adapter,
            device,
            queue,
        })
    }
}

/// Struct for managing all of the machines devices
pub struct DevicePool {
    instance: wgpu::Instance,
    pub devices: Vec<Device>
}


impl DevicePool {
    /// Creates a new device pool with a gpu
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let mut devices: Vec<Device> = Vec::new();
        devices.push(
            Device::new_gpu(&instance).await?
        );
        Ok(Self {
            instance,
            devices,
        })
    }
}
