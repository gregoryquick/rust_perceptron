use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
//use futures::executor::block_on;


#[derive(Serialize, Deserialize, Debug)]
pub struct Batchnorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub dimension: usize,
}

#[typetag::serde]
impl super::NetworkLayer for Batchnorm {
    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(2);
        
        let layer_gamma: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.gamma[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(layer_gamma);
        
        let layer_beta: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.beta[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(layer_beta);

        return vec;
    }

    fn forward(&self, 
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        let mut gpu_data = layer_data.iter();
        let layer_gamma = gpu_data.next().unwrap();
        let layer_beta = gpu_data.next().unwrap();

        //Create mean pipeline
        let mean_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };

        let mean_pipeline = pipelines::batchmean::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run weight pipeline
        mean_pipeline.run(encoder, self.dimension, batch_size);

        //Create variance pipeline
        let variance_pipeline = pipelines::batchvar::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                input,
                &mean_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run weight pipeline
        variance_pipeline.run(encoder, self.dimension, batch_size);

        //Create normalization pipeline
        let normalization_pipeline = pipelines::batchnorm::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                input,
                &mean_pipeline.output_buffer,
                &variance_pipeline.output_buffer,
                layer_beta,
                layer_gamma,
            ),
            self.dimension,
            batch_size,
        );

        //Run normalization pipeline
        normalization_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        normalization_pipeline.output_buffer
    }
}

