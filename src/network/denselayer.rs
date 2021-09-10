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
    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(1);
        
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
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
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
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
                    usage: wgpu::BufferUsage::UNIFORM,
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
                    usage: wgpu::BufferUsage::UNIFORM,
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
               batch_size: usize,) -> Vec<wgpu::Buffer> {
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
                    usage: wgpu::BufferUsage::UNIFORM,
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
                    usage: wgpu::BufferUsage::UNIFORM,
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

        
        //Create vec for return
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(1);
        vec.push(activation_pipeline.output_buffer);

        //Return
        vec
    }
}

