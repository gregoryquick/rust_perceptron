use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use futures::executor::block_on;

#[derive(Serialize, Deserialize, Debug)]
pub struct Denselayer {
    pub weights: Vec<f32>,
    pub output_dimension: usize,
    pub input_dimension: usize,
}

#[typetag::serde]
impl super::NetworkLayer for Denselayer {
    fn get_topology(&self) -> Vec<(usize, usize)> {
        let mut vec: Vec<(usize, usize)> = Vec::with_capacity(1);
        vec.push((self.output_dimension, self.input_dimension));
        
        //Return
        vec
    }

    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(1);
        
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }
        );
        vec.push(layer_weights);
        
        return vec;
    }

    fn save_from_gpu(&mut self, anchor: &pipelines::Device, data: &Vec<wgpu::Buffer>) {
        let queue = &anchor.queue;
        let device = &anchor.device;
        let type_size = std::mem::size_of::<f32>();

        let mut gpu_data = data.into_iter();
        let layer_weights = gpu_data.next().unwrap();

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Copy to readable buffer
        let layer_weight_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.output_dimension *  self.input_dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            layer_weights, 0,
            &layer_weight_buffer, 0,
            (type_size * self.output_dimension *  self.input_dimension) as wgpu::BufferAddress,
        );

        //Submit commands to gpu
        queue.submit(Some(encoder.finish()));

        //Create future of the computation
        let layer_weight_slice = layer_weight_buffer.slice(..);
        let layer_weight_future = layer_weight_slice.map_async(wgpu::MapMode::Read);
        
        //Register mapping callbacks
        device.poll(wgpu::Maintain::Wait);

        //Read from gpu
        block_on(async {
            match layer_weight_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = layer_weight_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    layer_weight_buffer.unmap();

                    //Save data
                    self.weights = result;
                }
                Err(e) => {
                    eprintln!("Failed to save layer_weights to cpu: {}", e);
                }
            }
        });
    }

    fn forward(&self, 
               input: &wgpu::Buffer,
               layer_data: &Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> wgpu::Buffer {
        let device = &anchor.device;
        
        let mut gpu_data = layer_data.into_iter();
        let layer_weights = gpu_data.next().unwrap();

        //Create weight application pipeline
        let weight_uniforms = {
            let uniform_data = [self.output_dimension as u32, self.input_dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let weight_pipeline = pipelines::matrixmultiply::Pipeline::new::<f32>(anchor, (
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

        //Create activation pipeline
        let activation_uniforms = {
            let uniform_data = [self.output_dimension as u32, batch_size as u32,];
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
                &weight_pipeline.output_buffer,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, self.output_dimension, batch_size);

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
        
        let mut gpu_data = layer_data.into_iter();
        let layer_weights = gpu_data.next().unwrap();

        //Create weight application pipeline
        let weight_uniforms = {
            let uniform_data = [self.output_dimension as u32, self.input_dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let weight_pipeline = pipelines::matrixmultiply::Pipeline::new::<f32>(anchor, (
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

        //Create activation pipeline
        let activation_uniforms = {
            let uniform_data = [self.output_dimension as u32, batch_size as u32,];
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
                &weight_pipeline.output_buffer,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(encoder, self.output_dimension, batch_size);
        
        //Create activationprime pipeline
            let activationprime_pipeline = pipelines::leakyreluprime::Pipeline::new::<f32>(anchor, (
                &activation_uniforms,
                &weight_pipeline.output_buffer,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run activationprime pipeline
        activationprime_pipeline.run(encoder, self.output_dimension, batch_size);

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
        
        let mut gpu_data = layer_data.into_iter();
        let layer_weights = gpu_data.next().unwrap();

        let mut gpu_data = backprop_data.into_iter();
        let layer_outputprime = gpu_data.next().unwrap();
        let layer_input = gpu_data.next().unwrap();

        //Create backprop_error pipeline
        let backprop_error_uniforms = {
            let uniform_data = [self.output_dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let backprop_error_pipeline = pipelines::elementmultiply::Pipeline::new::<f32>(anchor, (
                &backprop_error_uniforms,
                layer_outputprime,
                backprop_grad,
            ),
            self.output_dimension,
            batch_size,
        );

        //Run backprop_error pipeline
        backprop_error_pipeline.run(encoder, self.output_dimension, batch_size);

        //Create weight_grad pipeline
        let weight_grad_uniforms = {
            let uniform_data = [self.output_dimension as u32, batch_size as u32, self.input_dimension as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let weight_grad_pipeline = pipelines::multiplybytranspose::Pipeline::new::<f32>(anchor, (
                &weight_grad_uniforms,
                &backprop_error_pipeline.output_buffer,
                layer_input,
            ),
            self.output_dimension,
            batch_size,
            self.input_dimension,
        );

        //Run weight_grad pipeline
        weight_grad_pipeline.run(encoder, self.output_dimension, batch_size, self.input_dimension);

        //Create input_grad pipeline
        let input_grad_uniforms = {
            let uniform_data = [self.input_dimension as u32, self.output_dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            )
        };

        let input_grad_pipeline = pipelines::multiplytransposewith::Pipeline::new::<f32>(anchor, (
                &input_grad_uniforms,
                layer_weights,
                &backprop_error_pipeline.output_buffer,
            ),
            self.input_dimension,
            self.output_dimension,
            batch_size,
        );

        //Run input_grad pipeline
        input_grad_pipeline.run(encoder, self.input_dimension,  self.output_dimension, batch_size);
        
        //Return
        let mut vec: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(1);
        vec.push(Some(weight_grad_pipeline.output_buffer));
        (input_grad_pipeline.output_buffer, vec)
    }
}

