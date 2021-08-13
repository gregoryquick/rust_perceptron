use crate::pipelines;
use crate::optimisers;

use rand::prelude::*;
use futures::executor::block_on;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::collections::VecDeque;


pub mod denselayer;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    sizes: Vec<usize>,
    layers: Vec<denselayer::Denselayer>,
}

impl NeuralNetwork {
    pub fn new(sizes: Vec<usize>,) -> Self {
        let mut rng = rand::thread_rng();
        use rand::distributions::Uniform;
        let dist = Uniform::new(-1.0,1.0);
        
        let mut layers = Vec::new();

        for (input_size, output_size) in 
        sizes.split_last().unwrap().1.iter().zip(sizes.split_first().unwrap().1) {
            layers.push(denselayer::Denselayer{
                weights:{
                    let mut vector: Vec<f32> = vec![0f32; *input_size * *output_size];
                    for num in vector.iter_mut() {
                        *num = rng.sample(dist);
                    }
                    vector
                },
                biases:{
                    let mut vector: Vec<f32> = vec![0f32; *output_size];
                    for num in vector.iter_mut() {
                        *num = rng.sample(dist);
                    }
                    vector
                },
            });
        }
        
        NeuralNetwork {
            sizes,
            layers,
        }
    }

    pub fn save(&self, filelocation: &str) {
        let file = File::create(filelocation).unwrap();
        bincode::serialize_into(&file, &self).unwrap();
    }

    pub fn load(filelocation: &str) -> Self {
        let file = File::open(filelocation).unwrap();
        let network: NeuralNetwork = bincode::deserialize_from(&file).unwrap();
        network
    }

    pub fn feedforward<T: bytemuck::Pod>(&self, input: &Vec<T>, batch_size: usize,) -> Option<Vec<T>> {
        //Connect to device
        let anchor = block_on(pipelines::Device::new());
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
        let layer_iterator = self.sizes.split_last().unwrap().1.iter()
        .zip(self.sizes.split_first().unwrap().1)
        .zip(self.layers.iter());

        let output_buffer = layer_iterator.fold(input_buffer, |buffer, (topology, layer)| {
            let (&input_size, &output_size) = topology;

            layer.forward::<T>(
                buffer,
                &anchor,
                &mut encoder,
                output_size,
                input_size,
                batch_size,
            )
        });
        
        //Create staging buffer for loading out of gpu
        let staging_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.sizes.split_last().unwrap().0 * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );

        //Copy out of gpu
        encoder.copy_buffer_to_buffer(
            &output_buffer, 0,
            &staging_buffer, 0,
            (type_size * self.sizes.split_last().unwrap().0 * batch_size) as wgpu::BufferAddress,
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

    pub fn backprop<T: bytemuck::Pod>(&mut self, optimiser: &mut optimisers::Stochasticgradientdescent, input: &Vec<T>, labels: &Vec<T>, batch_size: usize,) {
        //Connect to device
        let anchor = block_on(pipelines::Device::new());
        let device = &anchor.device;
        let queue = &anchor.queue;
        let type_size = std::mem::size_of::<T>();
        
        //Load input and labels to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let input_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        let label_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&labels[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Get feed forward output data
        let layer_iterator = self.sizes.split_last().unwrap().1.iter()
        .zip(self.sizes.split_first().unwrap().1)
        .zip(self.layers.iter());

        let type_size = std::mem::size_of::<T>();

        let intermediate_values: VecDeque<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> = layer_iterator.scan(input_buffer, |buffer, (topology, layer)| {
            let (&input_size, &output_size) = topology;

            let computation_buffer =  device.create_buffer( &wgpu::BufferDescriptor {
                label: Some("Computation buffer"),
                size: (type_size * input_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(
                buffer, 0,
                &computation_buffer, 0,
                (type_size * input_size * batch_size) as wgpu::BufferAddress,
            );

            let (output, outputprime, input) = layer.forward_for_backprop::<T>(
                computation_buffer,
                &anchor,
                &mut encoder,
                output_size,
                input_size,
                batch_size,
            );

            let computation_buffer =  device.create_buffer( &wgpu::BufferDescriptor {
                label: Some("Computation buffer"),
                size: (type_size * output_size * batch_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC ,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(
                &output, 0,
                &computation_buffer, 0,
                (type_size * output_size * batch_size) as wgpu::BufferAddress,
            );

            
            *buffer = computation_buffer;
            Some((output, outputprime, input))
        }).collect();

        let mut backprop_iterator = self.sizes.split_last().unwrap().1.iter()
        .zip(self.sizes.split_first().unwrap().1)
        .zip(intermediate_values.into_iter())
        .zip(self.layers.iter_mut()).rev();

        match backprop_iterator.next() {
            Some(((beginning_topology, beginning_data), begining_layer)) => {
                let initial_step = {
                    let (&data_dimension, &output_dimension) = beginning_topology;

                    let (layer_activation, layer_activationprime, layer_input) = beginning_data;

                    //Create loss pipeline
                    let loss_pipeline = pipelines::elementsubtract::Pipeline::new::<T>(&anchor, (
                            None,
                            Some(layer_activation),
                            Some(label_buffer),
                            None,
                        ),
                        output_dimension,
                        batch_size,
                    );

                    //Run sensitivity pipeline
                    loss_pipeline.run(&anchor, &mut encoder, output_dimension, batch_size);
                
                    //Submit encoder
                    queue.submit(Some(encoder.finish()));

                    //Wait for computation to complete
                    device.poll(wgpu::Maintain::Wait);
                    begining_layer.backprop::<T>(
                        &anchor,
                        optimiser,
                        layer_input,
                        loss_pipeline.output_buffer,
                        layer_activationprime,
                        output_dimension,
                        data_dimension,
                        batch_size,
                    )
                };

                backprop_iterator.fold(initial_step, |intermediate_buffer, ((topology, data), layer)| {
                    let (&data_dimension, &output_dimension) = topology;

                    let (layer_activation, layer_activationprime, layer_input) = data;

                    layer.backprop::<T>(
                        &anchor,
                        optimiser,
                        layer_input,
                        intermediate_buffer,
                        layer_activationprime,
                        output_dimension,
                        data_dimension,
                        batch_size,
                    )
                });
            },
            None => {
                println!("Backprop Failed: No layers")
            }
        }
    }
}
