pub mod matrixdot;

pub struct PipelineAnchor {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_size: usize,
    output_size: usize,
}

impl PipelineAnchor {
    pub async fn new(input_size: usize, output_size: usize,) -> Self {
        //Get device
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
        
        //Return
        PipelineAnchor {
            adapter,
            device,
            queue,
            input_size,
            output_size,
        }
    }
}

