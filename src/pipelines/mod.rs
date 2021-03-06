pub mod addvectortobatch;
pub mod batchmax;
pub mod batchmean;
pub mod batchnorm;
pub mod batchnormprime;
pub mod batchtotal;
pub mod batchvar;
pub mod copymatrix;
pub mod crossentropy;
pub mod crossentropyprime;
pub mod dividebatchbyvector;
pub mod elementmultiply;
pub mod elementsubtract;
pub mod expfunct;
pub mod leakyrelu;
pub mod leakyreluprime;
pub mod matrixmultiply;
pub mod multiplybytranspose;
pub mod multiplytransposewith;
pub mod scalarmultiply;
pub mod scalebatchwithvector;
pub mod softmaxprime;
pub mod squarederror;
pub mod subtractscalarsfrombatch;
pub mod totalofbatch;
pub mod updatemean;
pub mod updatesample;
pub mod updatevar;

pub struct Device {
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Device {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
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
