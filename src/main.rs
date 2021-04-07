use rand::prelude::*;

use futures::executor::block_on;

fn main() {
    //Network parameters
    const DATA_DIM: usize = 16;
    const OUTPUT_DIM: usize = 2;

    //Make gpu pipelines
    let mut gpu_pipeline = block_on(PipelineManager::new(DATA_DIM, OUTPUT_DIM));
    let forward_pipeline = gpu_pipeline.new_pipeline::<ForwardPass, f64>();
    
    //Make network weights and a test input
    let mut rng = rand::thread_rng();
    let network_weights = {
        const WEIGHT_SIZE: usize = DATA_DIM * OUTPUT_DIM;
        let mut vector: [f64; WEIGHT_SIZE] = [0f64; WEIGHT_SIZE];
        for num in vector.iter_mut() {
            *num = rng.gen();
        }
        vector
    };
    let input_vector = {
        let mut vector: [f64; DATA_DIM] = [0f64; DATA_DIM];
        for num in vector.iter_mut() {
            *num = rng.gen();
        }
        vector
    };
    
    //Compute forward pass result
    let result = block_on(gpu_pipeline.run_forward_pass::<f64>(forward_pipeline, &network_weights, &input_vector)).unwrap();
    println!("{:?}", result);
}

struct PipelineManager {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    network_shape: (usize, usize),
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
            network_shape: (input_size, output_size),
        }
    }

    fn new_pipeline<P: Pipeline, T: bytemuck::Pod>(&self) -> P {
        P::new::<T>(&self.device, self.network_shape.0, self.network_shape.1)
    }

    async fn run_forward_pass<T: bytemuck::Pod>(&self, pipeline: ForwardPass, network_weights: &[T], input_vector: &[T]) -> Option<Vec<T>> {
        let type_size = std::mem::size_of::<T>() as wgpu::BufferAddress;

        //Create command encoder
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Load data into gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let buffers = pipeline.get_buffers();

        let weight_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(network_weights),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &weight_data_buffer, 0,
            &buffers[0], 0,
            (type_size * self.network_shape.0 as u64 * self.network_shape.1 as u64) as wgpu::BufferAddress,
        );
        
        let input_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(input_vector),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &weight_data_buffer, 0,
            &buffers[1], 0,
            (type_size * self.network_shape.0 as u64) as wgpu::BufferAddress,
        );

        //Create the compute pass (Mutably borrows encoder)
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { 
                label: None 
            }
        );
        
        //Add compute pipeline to compute pass
        compute_pass.set_pipeline(pipeline.get_compute_pipeline());
        compute_pass.set_bind_group(0, pipeline.get_bind_group(), &[]);
        //Work groups of x=output_size, Y = 1, Z = 1
        compute_pass.dispatch(self.network_shape.1 as u32, 1, 1);
        
        //Encoder borrow is gone now!
        drop(compute_pass);

        //Copy output from gpu buffer to staging buffer on cpu
        let staging_buffer = self.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.network_shape.1 as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            &buffers[2], 0,
            &staging_buffer, 0,
            (type_size * self.network_shape.1 as u64) as wgpu::BufferAddress,
        );

        //Finish building command encoder and submit
        self.queue.submit(Some(encoder.finish()));

        //Create future of the computation
        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        //Wait for computation to complete
        self.device.poll(wgpu::Maintain::Wait);

        //Handle computation result and return
        match buffer_future.await {
            Ok(()) => {
                //Get buffer contents
                let data = buffer_slice.get_mapped_range();
                let result: Vec<T> = data.chunks_exact(type_size as usize).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                
                //Drop mapped view
                drop(data);
                //Unmap buffer
                staging_buffer.unmap();

                Some(result)
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                None
            }
        }
    }
}

trait Pipeline {
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, input_size: usize, output_size: usize,) -> Self;

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
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, input_size: usize, output_size: usize,) -> Self{

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
        buffers.push(weight_buffer);

        let input_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Input Buffer"),
                size: (type_size * input_size as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        buffers.push(input_buffer);
        
        let output_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                size: (type_size * output_size as u64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        buffers.push(output_buffer);
        
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
                    resource: buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers[2].as_entire_binding(),
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
