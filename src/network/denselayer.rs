use crate::pipelines;

use futures::executor::block_on;
use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};


#[derive(Serialize, Deserialize, Debug)]
pub struct Denselayer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub output_dimension: usize,
    pub input_dimension: usize,
}

impl Denselayer {
    pub fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(2);
        
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(layer_weights);
        
        let layer_biases: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.biases[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(layer_biases);

        return vec;
    }

    pub fn forward<T: bytemuck::Pod>(&self, 
                                     input: &wgpu::Buffer,
                                     layer_data: &Vec<wgpu::Buffer>,
                                     anchor: &pipelines::Device,
                                     encoder: &mut wgpu::CommandEncoder,
                                     batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        let mut gpu_data = layer_data.iter();
        let layer_weights = gpu_data.next().unwrap();
        let layer_biases = gpu_data.next().unwrap();

        //Create weight application pipeline
        let weight_uniforms = {
            let uniform_data = [self.output_dimension as u32, self.input_dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };

        let weight_pipeline = pipelines::matrixmultiply::Pipeline::new::<T>(anchor, (
                &weight_uniforms,
                layer_weights,
                input,
            ),
            self.output_dimension,
            self.input_dimension,
            batch_size,
        );

        //Run weight pipeline
        weight_pipeline.run(encoder, self.output_dimension, self.input_dimension, batch_size);

        //Create bias pipeline
        let bias_uniforms = {
            let uniform_data = [self.output_dimension as u32, batch_size as u32,];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };
        
        let bias_pipeline = pipelines::addvectortobatch::Pipeline::new::<T>(anchor, (
                &bias_uniforms,
                &weight_pipeline.output_buffer,
                layer_biases,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run bias pipeline
        bias_pipeline.run(encoder, self.output_dimension, batch_size);

        //Create activation pipeline
        let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<T>(anchor, (
                &bias_uniforms,
                &bias_pipeline.output_buffer,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, self.output_dimension, batch_size);

        //Return
        activation_pipeline.output_buffer
    }
}

