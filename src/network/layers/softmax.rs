use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct Softmax {
    pub dimension: usize,
}

#[typetag::serde]
impl super::NetworkLayer for Softmax {
    fn get_topology(&self) -> Vec<(usize, usize)> {
        #[allow(unused_mut)]
        let mut vec: Vec<(usize, usize)> = Vec::with_capacity(0);
        
        //Return
        vec
    }

    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let _device = &anchor.device;
        #[allow(unused_mut)]
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(0);
        
        //Return
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

        //Create batchmax pipeline
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
        
        let batchmax_pipeline = pipelines::batchmax::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run batchmax pipeline
        batchmax_pipeline.run(encoder, self.dimension, batch_size);

        //Create batchshift pipeline
        let batchshift_pipeline = pipelines::subtractscalarsfrombatch::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
                &batchmax_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run batchshift pipeline
        batchshift_pipeline.run(encoder, self.dimension, batch_size);

        //Create exponential pipeline
        let exponential_pipeline = pipelines::expfunct::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &batchshift_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run exponential pipeline
        exponential_pipeline.run(encoder, self.dimension, batch_size);

        //Create denominator pipeline
        let denominator_pipeline = pipelines::totalofbatch::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &exponential_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run denominator pipeline
        denominator_pipeline.run(encoder, self.dimension, batch_size);

        //Run softmax pipeline
        let softmax_pipeline = pipelines::dividebatchbyvector::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &exponential_pipeline.output_buffer,
                &denominator_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run softmax pipeline
        softmax_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        softmax_pipeline.output_buffer
    }

    fn forward_for_backprop(&self, 
               input: &wgpu::Buffer,
               layer_data: &mut Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> (wgpu::Buffer, Vec<wgpu::Buffer>) {
        let device = &anchor.device;
        
        let mut _gpu_data = layer_data.into_iter();

        //Create batchmax pipeline
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
        
        let batchmax_pipeline = pipelines::batchmax::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
            ),
            self.dimension,
            batch_size,
        );

        //Run batchmax pipeline
        batchmax_pipeline.run(encoder, self.dimension, batch_size);

        //Create batchshift pipeline
        let batchshift_pipeline = pipelines::subtractscalarsfrombatch::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                input,
                &batchmax_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run batchshift pipeline
        batchshift_pipeline.run(encoder, self.dimension, batch_size);

        //Create exponential pipeline
        let exponential_pipeline = pipelines::expfunct::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &batchshift_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run exponential pipeline
        exponential_pipeline.run(encoder, self.dimension, batch_size);

        //Create denominator pipeline
        let denominator_pipeline = pipelines::totalofbatch::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &exponential_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run denominator pipeline
        denominator_pipeline.run(encoder, self.dimension, batch_size);

        //Run softmax pipeline
        let softmax_pipeline = pipelines::dividebatchbyvector::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &exponential_pipeline.output_buffer,
                &denominator_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        //Run softmax pipeline
        softmax_pipeline.run(encoder, self.dimension, batch_size);

        //Create a copy of the softmax buffer for backprop use
        let copy_pipeline = pipelines::copymatrix::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &softmax_pipeline.output_buffer,
            ),
            self.dimension,
            batch_size,
        );

        copy_pipeline.run(encoder, self.dimension, batch_size);

        //Create vec for return
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(2);
        vec.push(copy_pipeline.output_buffer);

        //Return
        (softmax_pipeline.output_buffer, vec)
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
        let layer_output = gpu_data.next().unwrap();
        let _layer_input = gpu_data.next().unwrap();

        //Create backprop_error pipeline
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

        let input_grad_pipeline = pipelines::softmaxprime::Pipeline::new::<f32>(anchor, (
                &input_grad_uniforms,
                layer_output,
                backprop_grad,
            ),
            self.dimension,
            batch_size,
        );

        //Run backprop_error pipeline
        input_grad_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        let mut vec: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(1);
        vec.push(None);
        (input_grad_pipeline.output_buffer, vec)
    }
}

