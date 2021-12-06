use std::error::Error;

pub struct DevicePool {
    cpu: CPU,
    gpus: Vec<GPU>,
}

impl DevicePool {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let mut gpus: Vec<GPU> = Vec::new();
        
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).map(make_gpu);
        while let Some(gpu) = adapters.next() {
            gpus.push(gpu.await?);
        }
        Ok(Self {
            cpu: CPU {},
            gpus,
        })
    }

    pub fn cpu(&self) -> &CPU {
        &self.cpu
    }

    pub fn gpus(&self) -> Vec<&GPU> {
        self.gpus.iter().collect()
    }
}



///
pub trait Device {
}

#[derive(Debug)]
pub struct GPU {
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

async fn make_gpu(adapter: wgpu::Adapter,) -> Result<GPU, Box<dyn Error>> {
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },
        None,
    ).await?;
    
    Ok(GPU {
        adapter,
        device,
        queue,
    })
}

impl Device for GPU {
}

pub struct CPU;

impl Device for CPU {
}

