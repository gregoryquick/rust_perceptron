pub mod matrixmultiply;

pub struct Device {
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Device {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();

        Device {
            _adapter: adapter,
            device,
            queue,
        }
    }
}
