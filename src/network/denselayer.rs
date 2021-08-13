use crate::pipelines;
use crate::optimisers;

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
                Some(bias_pipeline.uniform_buffer),
                Some(bias_pipeline.output_buffer),
                None,
            ),
            output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(anchor, encoder, output_dimension, batch_size);

        //Return
        activation_pipeline.output_buffer
    }

    pub fn forward_for_backprop<T: bytemuck::Pod>(&self, 
                                                  input: wgpu::Buffer,
                                                  anchor: &pipelines::Device,
                                                  encoder: &mut wgpu::CommandEncoder,
                                                  output_dimension: usize,
                                                  data_dimension: usize,
                                                  batch_size: usize,) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        
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
                Some(bias_pipeline.uniform_buffer),
                Some(bias_pipeline.output_buffer),
                None,
            ),
            output_dimension,
            batch_size,
        );

        //Run activation pipeline
        activation_pipeline.run(anchor, encoder, output_dimension, batch_size);

        //Create activationprime pipeline
        let activationprime_pipeline = pipelines::leakyreluprime::Pipeline::new::<T>(anchor, (
                Some(activation_pipeline.uniform_buffer),
                Some(activation_pipeline.matrix_buffer),
                None,
            ),
            output_dimension,
            batch_size,
        );
                
        //Run activationprime pipeline
        activationprime_pipeline.run(anchor, encoder, output_dimension, batch_size);

        //Return
        (activation_pipeline.output_buffer, activationprime_pipeline.output_buffer, weight_pipeline.matrix_b_buffer)
    }

    pub fn backprop<T: bytemuck::Pod>(&mut self,
                                      anchor: &pipelines::Device,
                                      optimiser: &mut optimisers::Stochasticgradientdescent,
                                      input: wgpu::Buffer,
                                      output: wgpu::Buffer,
                                      outputprime: wgpu::Buffer,
                                      output_dimension: usize,
                                      data_dimension: usize,
                                      batch_size: usize,) -> wgpu::Buffer {
        
        let device = &anchor.device;
        let queue = &anchor.queue;
        let type_size = std::mem::size_of::<T>();

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Create sensitivity pipeline
        let sensitivity_pipeline = pipelines::elementmultiply::Pipeline::new::<T>(anchor, (
                None,
                Some(output),
                Some(outputprime),
                None,
            ),
            output_dimension,
            batch_size,
        );

        //Run sensitivity pipeline
        sensitivity_pipeline.run(anchor, &mut encoder, output_dimension, batch_size);

        //Create gradient pipeline
        let gradient_pipeline = pipelines::multiplybytranspose::Pipeline::new::<T>(anchor, (
                None,
                Some(sensitivity_pipeline.output_buffer),
                Some(input),
                None,
            ),
            output_dimension,
            batch_size,
            data_dimension,
        );

        //Run gradient pipeline
        gradient_pipeline.run(anchor, &mut encoder, output_dimension, batch_size, data_dimension);

        //Load layer weight data to gpu
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let layer_weights: wgpu::Buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&self.weights[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        //Create intermediate pipeline
        let intermediate_pipeline = pipelines::multiplytransposewith::Pipeline::new::<T>(anchor, (
                None,
                Some(layer_weights),
                Some(gradient_pipeline.matrix_a_buffer),
                None,
            ),
            data_dimension,
            output_dimension,
            batch_size,
        );

        //Run intermediate pipeline
        intermediate_pipeline.run(anchor, &mut encoder, data_dimension, output_dimension, batch_size);

        //Use optimiser to get new weights
        let new_weights = optimiser.partial_step(
            anchor,
            &mut encoder,
            intermediate_pipeline.matrix_a_buffer,
            gradient_pipeline.output_buffer,
            output_dimension,
            data_dimension,
        );

        //Create staging buffer for loading out of gpu
        let staging_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * output_dimension * data_dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );

        //Copy out of gpu
        encoder.copy_buffer_to_buffer(
            &new_weights, 0,
            &staging_buffer, 0,
            (type_size * output_dimension * data_dimension) as wgpu::BufferAddress,
        );

        //Submit encoder
        queue.submit(Some(encoder.finish()));

        //Create future of the new weights
        let buffer_slice = staging_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        
        //Wait for computation to complete
        device.poll(wgpu::Maintain::Wait);

        //Update weights
        block_on(async {
            match buffer_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = buffer_slice.get_mapped_range();
                    //Convert to T
                    let new_weights: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    staging_buffer.unmap();

                    self.weights = new_weights;
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        });

        //Return
        intermediate_pipeline.output_buffer
    }
    
}

