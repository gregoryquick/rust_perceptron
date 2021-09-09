use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
//use futures::executor::block_on;


#[derive(Serialize, Deserialize, Debug)]
pub struct Batchnorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub data_var: Vec<f32>,
    pub data_mean: Vec<f32>,
    pub batches_sampled: u32,
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

        let data_var: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.data_var[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(data_var);

        let data_mean: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.data_mean[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(data_mean);

        let batches_sampled: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[self.batches_sampled]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );
        vec.push(batches_sampled);

        return vec;
    }

    fn forward(&self, 
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        let mut gpu_data = layer_data.into_iter();
        let layer_gamma = gpu_data.next().unwrap();
        let layer_beta = gpu_data.next().unwrap();
        let data_var =  gpu_data.next().unwrap();
        let data_mean = gpu_data.next().unwrap();
        let _batches_sampled = gpu_data.next().unwrap();

        //Create normalization pipeline
        let normalization_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };

        let normalization_pipeline = pipelines::batchnorm::Pipeline::new::<f32>(anchor, (
                &normalization_uniforms,
                input,
                data_mean,
                data_var,
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

    fn forward_for_backprop(&self, 
               input: &wgpu::Buffer,
               layer_data: &mut Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        
        let mut gpu_data = layer_data.into_iter();
        let layer_gamma = gpu_data.next().unwrap();
        let layer_beta = gpu_data.next().unwrap();
        let data_var =  gpu_data.next().unwrap();
        let data_mean = gpu_data.next().unwrap();
        let batches_sampled = gpu_data.next().unwrap();

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

        //Run mean pipeline
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

        //Run variance pipeline
        variance_pipeline.run(encoder, self.dimension, batch_size);

        //Create sample update pipeline
        let sample_update_pipeline = pipelines::updatesample::Pipeline::new::<f32>(anchor, batches_sampled,);

        //Run sample update pipeline
        sample_update_pipeline.run(encoder);

        //Create var update pipeline
        let var_update_uniforms = {
            let uniform_data = [self.dimension as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };

        let var_update_pipeline = pipelines::updatevar::Pipeline::new::<f32>(anchor, (
                &var_update_uniforms,
                data_var,
                &variance_pipeline.output_buffer,
                data_mean,
                &mean_pipeline.output_buffer,
                &sample_update_pipeline.output_buffer,
            ),
            self.dimension,
        );

        //Run mean update pipeline
        var_update_pipeline.run(encoder, self.dimension);

        //Create mean update pipeline
        let mean_update_pipeline = pipelines::updatemean::Pipeline::new::<f32>(anchor, (
                &var_update_uniforms,
                data_mean,
                &mean_pipeline.output_buffer,
                &sample_update_pipeline.output_buffer,
            ),
            self.dimension,
        );

        //Run mean update pipeline
        mean_update_pipeline.run(encoder, self.dimension);

        //Create normalization pipeline
        let normalization_pipeline = pipelines::batchnorm::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                input,
                &mean_update_pipeline.output_buffer,
                &var_update_pipeline.output_buffer,
                layer_beta,
                layer_gamma,
            ),
            self.dimension,
            batch_size,
        );

        //Run normalization pipeline
        normalization_pipeline.run(encoder, self.dimension, batch_size);

        //Create vec for return
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(1);
        vec.push(normalization_pipeline.output_buffer);

        //Update mutable values
        *data_mean = mean_update_pipeline.output_buffer;
        *data_var = var_update_pipeline.output_buffer;
        *batches_sampled = sample_update_pipeline.output_buffer;
        
        //Return
        vec
    }

}

