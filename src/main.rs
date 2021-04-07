use rand::prelude::*;

use futures::executor::block_on;

fn main() {
    const DATA_DIM: usize = 16;
    const OUTPUT_DIM: usize = 2;
    let mut rng = rand::thread_rng();
    let mut gpu_pipeline = block_on(PipelineManager::new(DATA_DIM, OUTPUT_DIM));
    let random_data = create_random_array::<DATA_DIM>(&mut rng);
    println!("{:?}", random_data);
}

//How to make this generic over datatype?
fn create_random_array<const SIZE: usize>(rng: &mut rand::rngs::ThreadRng) -> [f64; SIZE] {
    let mut vector: [f64; SIZE] = [0f64; SIZE];
    for num in vector.iter_mut() {
        *num = rng.gen();
    }
    vector
}

struct PipelineManager {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    network_shape: (usize, usize),
    //network_buffer: wgpu::Buffer,
}

impl PipelineManager{
    async fn new(input_size: usize, output_size: usize,) -> Self{
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
        
        PipelineManager{
            adapter: adapter,
            device: device,
            queue: queue,
            network_shape: (input_size,output_size),
        }
    }
}

trait Pipeline {
    fn new<T: bytemuck::Pod>(device: wgpu::Device, input_size: usize, output_size: usize,) -> Self;

    fn get_buffers(&self) -> &Vec<wgpu::Buffer>;

    fn get_bind_group(&self) -> &wgpu::BindGroup;

    fn get_compute_pipeline(&self) -> &wgpu::ComputePipeline;
}

struct ForwardPass {
    buffers: Vec<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl Pipeline for ForwardPass {
    fn new<T: bytemuck::Pod>(device: wgpu::Device, input_size: usize, output_size: usize,) -> Self{

        //Create buffers
        let mut buffers: Vec<wgpu::Buffer> = Vec::new();
        let type_size = std::mem::size_of::<T>() as wgpu::BufferAddress;

        let weight_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Network Weights"),
                size: (type_size * input_size as u64 * output_size as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        //buffers.push(weight_buffer);

        let input_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Input Buffer"),
                size: (type_size * input_size as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        //buffers.push(input_buffer);
        
        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * output_size as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        //buffers.push(output_buffer);
        
        //Create buffer bind group for pipeline
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(0),
                    },
                    count: None,
                },],
            }
        );
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Forward Pass bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },],
            }
        );

        //Create compute pipeline
        let cs_src = include_str!("shaders/forward.comp");
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_spirv = compiler.compile_into_spirv(cs_src, shaderc::ShaderKind::Compute, "shader.comp", "main", None).unwrap();
        let cs_module = device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::util::make_spirv(cs_spirv.as_binary_u8()),
                flags: wgpu::ShaderFlags::empty(),
            }
        );
        
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );
        
        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            }
        );
        ForwardPass{
            buffers,
            bind_group,
            compute_pipeline,
        }
    }

    fn get_buffers(&self) -> &Vec<wgpu::Buffer>{
        &self.buffers
    }

    fn get_bind_group(&self) -> &wgpu::BindGroup{
        &self.bind_group
    }

    fn get_compute_pipeline(&self) -> &wgpu::ComputePipeline{
        &self.compute_pipeline
    }
}

//<T: bytemuck::Pod>(input_size: usize, output_size: usize,)
