use crate::pipelines;

use rand::prelude::*;
use futures::executor::block_on;
use serde::{Serialize, Deserialize};
use std::fs::File;
//use std::collections::VecDeque;


pub mod denselayer;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<denselayer::Denselayer::<f32>>,
    //layers: Vec<Box<dyn NetworkLayer<f32>>>,
    output_size: usize,
}

pub trait NetworkLayer<T: bytemuck::Pod> {
    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer>;
    
    fn forward(&self,
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer;
}

impl NeuralNetwork {
    pub fn new(sizes: Vec<usize>,) -> Self {
        let mut rng = rand::thread_rng();
        use rand::distributions::Uniform;
        let dist = Uniform::new(-1.0,1.0);
        
        //let mut layers: Vec<Box<dyn NetworkLayer<f32>>> = Vec::new();
        let mut layers: Vec<denselayer::Denselayer::<f32>> = Vec::new();

        for (input_size, output_size) in 
        sizes.split_last().unwrap().1.iter().zip(sizes.split_first().unwrap().1) {
            //layers.push(Box::new(
            //        denselayer::Denselayer {
            //            weights:{
            //                let vector: Vec<f32> = (0..*input_size * *output_size).map(|_i| {rng.sample(dist)}).collect();
            //                vector
            //            },
            //            biases:{
            //                let vector: Vec<f32> = (0..*output_size).map(|_i| {rng.sample(dist)}).collect();
            //                vector
            //            },
            //            output_dimension: *output_size,
            //            input_dimension: *input_size,
            //        }
            //    )
            //);
            layers.push(denselayer::Denselayer {
                    weights:{
                        let vector: Vec<f32> = (0..*input_size * *output_size).map(|_i| {rng.sample(dist)}).collect();
                        vector
                    },
                    biases:{
                        let vector: Vec<f32> = (0..*output_size).map(|_i| {rng.sample(dist)}).collect();
                        vector
                    },
                    output_dimension: *output_size,
                    input_dimension: *input_size,
                }
            );
        }
        
        NeuralNetwork {
            layers,
            output_size: *sizes.split_last().unwrap().0,
        }
    }

    pub fn save_to_file(&self, filelocation: &str) {
        let file = File::create(filelocation).unwrap();
        bincode::serialize_into(&file, &self).unwrap();
    }

    pub fn load_from_file(filelocation: &str) -> Self {
        let file = File::open(filelocation).unwrap();
        let network: NeuralNetwork = bincode::deserialize_from(&file).unwrap();
        network
    }
    
    pub fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<Vec<wgpu::Buffer>> {
        let mut vec: Vec<Vec<wgpu::Buffer>> = Vec::new();
        for layer in &self.layers {
            vec.push(layer.load_to_gpu(anchor));
         }
        vec
    }

    pub fn feedforward<T: bytemuck::Pod>(&self,
                                         input: &Vec<T>,
                                         network_data: &Vec<Vec<wgpu::Buffer>>,
                                         anchor: &pipelines::Device,
                                         batch_size: usize,) -> Option<Vec<T>> {
        let queue = &anchor.queue;
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

        //Feed input through layers to get output
        let layer_iterator = self.layers.iter().zip(network_data.iter());
        let output_buffer = layer_iterator.fold(input_buffer, |buffer, (layer, layer_data)| {
            layer.forward(
                &buffer,
                layer_data,
                &anchor,
                &mut encoder,
                batch_size,
            )
        });
        
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
