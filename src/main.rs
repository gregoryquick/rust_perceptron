mod pipelines;

use rand::prelude::*;

use futures::executor::block_on;
use std::thread;

fn main() {
    //Default stack size is 8 * 1024 * 1024
    //All sizes for batches assum 256*256 f32
    //Batch size 4 needs >= 9 * 1024 * 1024
    //Batch size 8 needs >= 11 * 1024 * 1024
    //Batch size 32 needs >= 23 * 1024 * 1024
    //Batch size 64 needs >= 39 * 1024 * 1024
    //Batch size 128 needs >= 71 * 1024 * 1024
    //Batch size 256 needs >= 135 * 1024 * 1024
    //Batch size 512 needs >= 263 * 1024 * 1024

    const STACK_SIZE: usize = 263 * 1024 * 1024;    
    // Spawn thread with explicit stack size
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}

fn run() {
    //Network parameters
    const DATA_DIM: usize = 65536;
    const OUTPUT_DIM: usize = 12;

    //Make gpu pipelines
    let pipeline_manager = block_on(PipelineManager::new(DATA_DIM, OUTPUT_DIM));
    
    //Make network weights and a test input
    use rand::distributions::{Distribution, Uniform};
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-1.0,1.0);
    let network_weights = {
        const WEIGHT_SIZE: usize = DATA_DIM * OUTPUT_DIM;
        let mut vector: [f32; WEIGHT_SIZE] = [0f32; WEIGHT_SIZE];
        for num in vector.iter_mut() {
            *num = rng.sample(dist);
        }
        vector
    };
    //println!("Weights:");
    //println!("{:?}", network_weights);
    
        
    const BATCH_SIZE: usize = 512;
    let input_vector = {
        const DATA_SIZE: usize = DATA_DIM * BATCH_SIZE;
        let mut vector: [f32; DATA_SIZE] = [0f32; DATA_SIZE];
        for num in vector.iter_mut() {
            *num = rng.sample(dist);
        }
        vector
    };
    //println!("Input:");
    //println!("{:?}", input_vector);
    
    //Compute forward pass result
    let forward_pipeline = pipeline_manager.new_pipeline::<pipelines::ForwardPass, f32>(BATCH_SIZE);

    let result = block_on(pipeline_manager.run_forward_pass::<f32>(forward_pipeline, &network_weights, &input_vector)).unwrap();
    println!("Result:");
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

    fn new_pipeline<P: pipelines::Pipeline, T: bytemuck::Pod>(&self, batch_size: usize,) -> P {
        P::new::<T>(&self.device, self.network_shape.0, self.network_shape.1, batch_size)
    }

    async fn run_forward_pass<T: bytemuck::Pod>(&self, pipeline: pipelines::ForwardPass, network_weights: &[T], input_vector: &[T]) -> Option<Vec<T>> {
        use pipelines::*;
        let type_size = std::mem::size_of::<T>();

        //Create command encoder
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Load data into gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let buffers = pipeline.get_buffers();
        let batch_size: usize = pipeline.get_batch_size();


        let weight_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(network_weights),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &weight_data_buffer, 0,
            &buffers[1], 0,
            (type_size * self.network_shape.0 * self.network_shape.1) as wgpu::BufferAddress,
        );
        
        let input_data_buffer = self.device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(input_vector),
                usage: wgpu::BufferUsage::COPY_SRC,
            }
        );
        encoder.copy_buffer_to_buffer(
            &input_data_buffer, 0,
            &buffers[2], 0,
            (type_size * self.network_shape.0 * batch_size) as wgpu::BufferAddress,
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
        //Work groups of x=output_size, Y = batch_size, Z = 1
        compute_pass.dispatch(self.network_shape.1 as u32, batch_size as u32, 1);
        
        //Encoder borrow is gone now!
        drop(compute_pass);

        //Copy output from gpu buffer to staging buffer on cpu
        let staging_buffer = self.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.network_shape.1 * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            &buffers[3], 0,
            &staging_buffer, 0,
            (type_size * self.network_shape.1 * batch_size) as wgpu::BufferAddress,
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
                let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                
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
