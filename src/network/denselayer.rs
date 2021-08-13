use crate::pipelines;

use futures::executor::block_on;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Denselayer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl Denselayer {
    pub fn forward<T: bytemuck::Pod>(&self, 
                                     input: wgpu::Buffer,
                                     anchor: &pipelines::Device,
                                     encoder: &mut wgpu::CommandEncoder,
                                     output_dimension: usize,
                                     data_dimension: usize,
                                     batch_size: usize,) -> wgpu::Buffer {
        
        let device = &anchor.device;
        
        //Load layer data to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        let layer_biases: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.biases[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        //Create weight application pipeline
        let weight_pipeline = pipelines::matrixmultiply::Pipeline::new::<T>(anchor, (
                None,
                Some(layer_weights),
                Some(input),
                None,
            ),
            output_dimension,
            data_dimension,
            batch_size,
        );

        //Run weight pipeline
        weight_pipeline.run(anchor, encoder, output_dimension, data_dimension, batch_size);

        //Create bias pipeline
        let bias_pipeline = pipelines::addvectortobatch::Pipeline::new::<T>(anchor, (
                None,
                Some(weight_pipeline.output_buffer),
                Some(layer_biases),
                None,
            ),
            output_dimension,
            batch_size,
        );

        //Run bias pipeline
        bias_pipeline.run(anchor, encoder, output_dimension, batch_size);

        //Create activation pipeline
        let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<T>(anchor, (
                &bias_pipeline.uniform_buffer,
                &bias_pipeline.output_buffer,
            ),
            output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, output_dimension, batch_size);

        //Return
        activation_pipeline.output_buffer
    }
}

