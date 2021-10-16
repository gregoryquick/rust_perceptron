use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct Relu {
    pub dimension: usize,
}

#[typetag::serde]
impl super::NetworkLayer for Relu {
    fn get_topology(&self) -> Vec<(usize, usize)> {
        #[allow(unused_mut)]
        let mut vec: Vec<(usize, usize)> = Vec::with_capacity(0);

        vec
    }

    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let _device = &anchor.device;
        #[allow(unused_mut)]
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(0);

        return vec;
    }

    fn save_from_gpu(&mut self, _anchor: &pipelines::Device, _data: &Vec<wgpu::Buffer>) {
        //Nothing to do
    }

    fn forward(&self, 
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        let mut _gpu_data = layer_data.into_iter();

        //Create activation pipeline
        let activation_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32,];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };
        
        let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        activation_pipeline.output_buffer
    }

    fn forward_for_backprop(&self, 
               input: &wgpu::Buffer,
               layer_data: &mut Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> (wgpu::Buffer, Vec<wgpu::Buffer>) {
        let device = &anchor.device;
        
        let mut _gpu_data = layer_data.into_iter();

        //Create activation pipeline
        let activation_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32,];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };
        
        let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, self.dimension, batch_size);
        
        //Create activationprime pipeline
        let activationprime_pipeline = pipelines::leakyreluprime::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run activationprime pipeline
        activationprime_pipeline.run(encoder, self.dimension, batch_size);

        //Create vec for return
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(2);
        vec.push(activationprime_pipeline.output_buffer);

        //Return
        (activation_pipeline.output_buffer, vec)
    }

    fn backprop(&self,
                backprop_grad: &wgpu::Buffer,
                layer_data: &Vec<wgpu::Buffer>, 
                backprop_data: &Vec<wgpu::Buffer>,
                anchor: &pipelines::Device,
                encoder: &mut wgpu::CommandEncoder,
                batch_size: usize,) -> (wgpu::Buffer, Vec<Option<wgpu::Buffer>>) {
        let device = &anchor.device;
        
        let mut _gpu_data = layer_data.into_iter();

        let mut gpu_data = backprop_data.into_iter();
        let layer_outputprime = gpu_data.next().unwrap();
        let _layer_input = gpu_data.next().unwrap();

        //Create input_grad pipeline
        let input_grad_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let input_grad_pipeline = pipelines::elementmultiply::Pipeline::new::<f32>(anchor, (
                &input_grad_uniforms,
                layer_outputprime,
                backprop_grad,
            ),
            self.dimension,
            batch_size,
        );

        //Run input_grad pipeline
        input_grad_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        let mut vec: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(1);
        vec.push(None);
        (input_grad_pipeline.output_buffer, vec)
    }
}

