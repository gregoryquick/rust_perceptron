use crate::pipelines;

use rand::prelude::*;
use futures::executor::block_on;

pub mod denselayer;

pub struct NeuralNetwork {
    input_size: usize,
    output_size: usize,
    layer: denselayer::Denselayer
}

impl NeuralNetwork {
    pub fn new(input_size: usize, output_size: usize) -> Self{
        let mut rng = rand::thread_rng();
        use rand::distributions::Uniform;
        let dist = Uniform::new(-1.0,1.0);
        
        let layer = denselayer::Denselayer{
            weights:{
                let mut vector: Vec<f32> = vec![0f32; input_size * output_size];
                for num in vector.iter_mut() {
                    *num = rng.sample(dist);
                }
                vector
            },
            biases:{
                 let mut vector: Vec<f32> = vec![0f32; output_size];
                 for num in vector.iter_mut() {
                    *num = rng.sample(dist);
                 }
                 vector
            },
        };
        
        NeuralNetwork{
            input_size,
            output_size,
            layer,
        }
    }

    pub fn feedforward<T: bytemuck::Pod>(self, input: Vec<T>, batch_size: usize,) -> Option<Vec<T>> {
        //Connect to device
        let anchor = block_on(pipelines::Device::new());
        let device = &anchor.device;
        let type_size = std::mem::size_of::<T>();
        
        //Load input to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let input_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );
        //Run through first layer
        let output_buffer = self.layer.forward::<T>(
            input_buffer,
            &anchor,
            &mut encoder,
            self.output_size,
            self.input_size,
            batch_size,
        );

        //Create staging buffer for loading out of gpu
        let staging_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );

        //Copy out of gpu
        encoder.copy_buffer_to_buffer(
            &output_buffer, 0,
            &staging_buffer, 0,
            (type_size * self.output_size * batch_size) as wgpu::BufferAddress,
        );

        //Submit encoder
        let queue = &anchor.queue;

        queue.submit(Some(encoder.finish()));

        //Create future of the computation
        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        
        //Wait for computation to complete
        device.poll(wgpu::Maintain::Wait);

        block_on(async {
            match buffer_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = buffer_slice.get_mapped_range();
                    //Convert to T
                    let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    staging_buffer.unmap();

                    return Some(result)
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    None
                }
            }
        })
    }
}
