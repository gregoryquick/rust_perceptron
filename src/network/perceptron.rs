use crate::pipelines;

use serde::{Serialize, Deserialize};
use std::fs::File;

use super::layers;
use super::cost;

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Box<dyn layers::NetworkLayer>>,
    cost_function: Box<dyn cost::CostFunction>,
    output_size: usize,
}

impl Network {
    pub fn new(input_size: usize, layer_types: Vec<super::LayerType>) -> Self {
        let mut layers: Vec<Box<dyn layers::NetworkLayer>> = Vec::new();
        let mut current_output: usize = input_size;

        for layer_type in layer_types.into_iter() {
            let (output_size, layer) = layers::generate_layer(current_output, layer_type);
            layers.push(layer);
            current_output = output_size;
        }
        
        let cost_function = cost::generate_cost(current_output, super::CostFunction::SquaredError);

        Network {
            layers,
            cost_function,
            output_size: current_output,
        }
    }

    pub fn get_topology(&self) -> Vec<Vec<(usize, usize)>> {
        let layer_iterator = self.layers.iter();
        let mut vec: Vec<Vec<(usize, usize)>> = Vec::new();
        for layer in layer_iterator {
            vec.push(layer.get_topology());
        }

        //Return
        vec
    }

    pub fn save_to_file(&self, filelocation: &str) {
        let file = File::create(filelocation).unwrap();
        bincode::serialize_into(&file, &self).unwrap();
    }

    pub fn load_from_file(filelocation: &str) -> Self {
        let file = File::open(filelocation).unwrap();
        let network: Network = bincode::deserialize_from(&file).unwrap();
        network
    }
    
    pub fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<Vec<wgpu::Buffer>> {
        let mut vec: Vec<Vec<wgpu::Buffer>> = Vec::new();
        for layer in &self.layers {
            vec.push(layer.load_to_gpu(anchor));
         }
        vec
    }

    pub fn save_from_gpu(&mut self, anchor: &pipelines::Device, data: &Vec<Vec<wgpu::Buffer>>) {
        let iter = self.layers.iter_mut().zip(data.into_iter());
        for (layer, layer_data) in iter {
            layer.save_from_gpu(anchor,layer_data);
        }
    }

    pub fn feedforward<T: bytemuck::Pod>(&self,
                                         input: &Vec<T>,
                                         network_data: &Vec<Vec<wgpu::Buffer>>,
                                         anchor: &pipelines::Device,
                                         batch_size: usize,) -> wgpu::Buffer {
        let queue = &anchor.queue;
        let device = &anchor.device;
        
        //Load input to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let input_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Feed input through layers to get output
        let layer_iterator = self.layers.iter().zip(network_data.into_iter());
        let output_buffer = layer_iterator.fold(input_buffer, |buffer, (layer, layer_data)| {
            layer.forward(
                &buffer,
                layer_data,
                &anchor,
                &mut encoder,
                batch_size,
            )
        });

        //Submit encoder
        queue.submit(Some(encoder.finish()));
        
        //Return
        output_buffer
    }

    pub fn cost<T: bytemuck::Pod>(&self,
                                  prediction: &wgpu::Buffer,
                                  labels: &Vec<T>,
                                  anchor: &pipelines::Device,
                                  batch_size: usize,) -> wgpu::Buffer {
        let queue = &anchor.queue;
        let device = &anchor.device;

        //Load data to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let label_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&labels[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );
        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Compute cost
        let cost = self.cost_function.cost(
            &prediction,
            &label_buffer,
            &anchor,
            &mut encoder,
            batch_size,
        );

        //Submit encoder
        queue.submit(Some(encoder.finish()));

        //Return
        cost
    }
    
    pub fn backprop<T: bytemuck::Pod>(&self,
                                      input: &Vec<T>,
                                      labels: &Vec<T>,
                                      network_data: &mut Vec<Vec<wgpu::Buffer>>,
                                      anchor: &pipelines::Device,
                                      batch_size: usize,) -> Vec<Vec<Option<wgpu::Buffer>>> {
        let queue = &anchor.queue;
        let device = &anchor.device;

        //Load data to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let input_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );
        
        let label_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&labels[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Feed input through layers to get info for backprop
        let layer_iterator = self.layers.iter().zip(network_data.iter_mut());
        let mut current_output = input_buffer;
        let intermediate_values = {
            let mut vec: Vec<Vec<wgpu::Buffer>> = Vec::new();
            for (layer, layer_data) in layer_iterator {
                let (new_output, mut data) = layer.forward_for_backprop(
                    &current_output,
                    layer_data,
                    &anchor,
                    &mut encoder,
                    batch_size,
                );
                data.push(current_output);
                vec.push(data);
                current_output = new_output;
            }
            vec
        };
        
        //Perform backprop
        let backprop_iter = self.layers.iter()
            .zip(network_data.iter())
            .zip(intermediate_values.iter())
            .rev();
        let mut backprop_grad = self.cost_function.cost_prime(
            &current_output,
            &label_buffer,
            &anchor,
            &mut encoder,
            batch_size,
        );
        
        let backprop_values = {
            let mut vec: Vec<Vec<Option<wgpu::Buffer>>> = Vec::new();
            for ((layer, layer_data), intermediate_data) in backprop_iter {
                let (layer_input_grad, layer_grads) = layer.backprop(
                    &backprop_grad,
                    layer_data,
                    intermediate_data,
                    &anchor,
                    &mut encoder,
                    batch_size,
                );
                vec.push(layer_grads);
                backprop_grad = layer_input_grad;
            }
            vec
        };

        //Submit encoder
        queue.submit(Some(encoder.finish()));

        //Return
        backprop_values
    }
}
