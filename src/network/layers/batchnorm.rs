use crate::pipelines;

use serde::{Serialize, Deserialize};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use futures::executor::block_on;

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
    fn get_topology(&self) -> Vec<(usize, usize)> {
        let mut vec: Vec<(usize, usize)> = Vec::with_capacity(5);
        vec.push((self.dimension, 1));
        vec.push((self.dimension, 1));
        vec.push((self.dimension, 1));
        vec.push((self.dimension, 1));
        vec.push((1, 1));
        
        //Return
        vec
    }

    fn load_to_gpu(&self, anchor: &pipelines::Device,) -> Vec<wgpu::Buffer> {
        let device = &anchor.device;
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(5);
        
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

    fn save_from_gpu(&mut self, anchor: &pipelines::Device, data: &Vec<wgpu::Buffer>) {
        let queue = &anchor.queue;
        let device = &anchor.device;
        let type_size = std::mem::size_of::<f32>();

        let mut gpu_data = data.into_iter();
        let layer_gamma = gpu_data.next().unwrap();
        let layer_beta = gpu_data.next().unwrap();
        let data_var =  gpu_data.next().unwrap();
        let data_mean = gpu_data.next().unwrap();
        let batches_sampled = gpu_data.next().unwrap();

        //Create command buffer encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None,
            }
        );

        //Copy to readable buffer
        let layer_gamma_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            layer_gamma, 0,
            &layer_gamma_buffer, 0,
            (type_size * self.dimension) as wgpu::BufferAddress,
        );

        let layer_beta_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            layer_beta, 0,
            &layer_beta_buffer, 0,
            (type_size * self.dimension) as wgpu::BufferAddress,
        );

        let data_var_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            data_var, 0,
            &data_var_buffer, 0,
            (type_size * self.dimension) as wgpu::BufferAddress,
        );

        let data_mean_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size * self.dimension) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            data_mean, 0,
            &data_mean_buffer, 0,
            (type_size * self.dimension) as wgpu::BufferAddress,
        );

        let batches_sampled_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging buffer"),
                size: (type_size) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }
        );
        encoder.copy_buffer_to_buffer(
            batches_sampled, 0,
            &batches_sampled_buffer, 0,
            (type_size) as wgpu::BufferAddress,
        );

        //Submit commands to gpu
        queue.submit(Some(encoder.finish()));

        //Create future of the data
        let layer_gamma_slice = layer_gamma_buffer.slice(..);
        let layer_gamma_future = layer_gamma_slice.map_async(wgpu::MapMode::Read);

        let layer_beta_slice = layer_beta_buffer.slice(..);
        let layer_beta_future = layer_beta_slice.map_async(wgpu::MapMode::Read);

        let data_var_slice = data_var_buffer.slice(..);
        let data_var_future = data_var_slice.map_async(wgpu::MapMode::Read);
        
        let data_mean_slice = data_mean_buffer.slice(..);
        let data_mean_future = data_mean_slice.map_async(wgpu::MapMode::Read);

        let batches_sampled_slice = batches_sampled_buffer.slice(..);
        let batches_sampled_future = batches_sampled_slice.map_async(wgpu::MapMode::Read);
        
        //Register mapping callbacks
        device.poll(wgpu::Maintain::Wait);

        //Read from gpu
        block_on(async {
            match layer_gamma_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = layer_gamma_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    layer_gamma_buffer.unmap();

                    //Save data
                    self.gamma = result;
                }
                Err(e) => {
                    eprintln!("Failed to save layer_gamma to cpu: {}", e);
                }
            }
        });

        block_on(async {
            match layer_beta_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = layer_beta_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    layer_beta_buffer.unmap();

                    //Save data
                    self.beta = result;
                }
                Err(e) => {
                    eprintln!("Failed to save layer_beta to cpu: {}", e);
                }
            }
        });

        block_on(async {
            match data_var_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = data_var_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    data_var_buffer.unmap();

                    //Save data
                    self.data_var = result;
                }
                Err(e) => {
                    eprintln!("Failed to save data_var to cpu: {}", e);
                }
            }
        });

        block_on(async {
            match data_mean_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = data_mean_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<f32> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<f32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    data_mean_buffer.unmap();

                    //Save data
                    self.data_mean = result;
                }
                Err(e) => {
                    eprintln!("Failed to save data_mean to cpu: {}", e);
                }
            }
        });

        block_on(async {
            match batches_sampled_future.await {
                Ok(()) => {
                    //Get buffer contents
                    let data = batches_sampled_slice.get_mapped_range();
                    //Convert to f32
                    let result: Vec<u32> = data.chunks_exact(std::mem::size_of::<u32>()).map(|b| *bytemuck::from_bytes::<u32>(b)).collect();
                     //Drop mapped view
                    drop(data);
                    //Unmap buffer
                    batches_sampled_buffer.unmap();

                    //Save data
                    self.batches_sampled = result[0];
                }
                Err(e) => {
                    eprintln!("Failed to save batches_sampled to cpu: {}", e);
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
            ),
            self.dimension,
            batch_size,
        );

        //Run normalization pipeline
        normalization_pipeline.run(encoder, self.dimension, batch_size);

        //Create scale pipeline
        let scale_pipeline = pipelines::scalebatchwithvector::Pipeline::new::<f32>(anchor, (
                &normalization_uniforms,
                &normalization_pipeline.output_buffer,
                layer_gamma,
            ),
            self.dimension,
            batch_size,
        );

        //Run scale pipeline
        scale_pipeline.run(encoder, self.dimension, batch_size);

        //Create bias pipeline
        let bias_pipeline = pipelines::addvectortobatch::Pipeline::new::<f32>(anchor, (
                &normalization_uniforms,
                &scale_pipeline.output_buffer,
                layer_beta,
            ),
            self.dimension,
            batch_size,
        );

        //Run bias pipeline
        bias_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        bias_pipeline.output_buffer
    }

    fn forward_for_backprop(&self, 
               input: &wgpu::Buffer,
               layer_data: &mut Vec<wgpu::Buffer>,
               anchor: &pipelines::Device,
               encoder: &mut wgpu::CommandEncoder,
               batch_size: usize,) -> (wgpu::Buffer, Vec<wgpu::Buffer>) {
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
            ),
            self.dimension,
            batch_size,
        );

        //Run normalization pipeline
        normalization_pipeline.run(encoder, self.dimension, batch_size);

        //Create batchnormprime pipeline
        let batchnormprime_pipeline = pipelines::batchnormprime::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                layer_gamma,
                data_var,
            ),
            self.dimension,
            batch_size,
        );

        //Run batchnormprime pipeline
        batchnormprime_pipeline.run(encoder, self.dimension, batch_size);
        
        //Create scale pipeline
        let scale_pipeline = pipelines::scalebatchwithvector::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                &normalization_pipeline.output_buffer,
                layer_gamma,
            ),
            self.dimension,
            batch_size,
        );

        //Run scale pipeline
        scale_pipeline.run(encoder, self.dimension, batch_size);

        //Create bias pipeline
        let bias_pipeline = pipelines::addvectortobatch::Pipeline::new::<f32>(anchor, (
                &mean_uniforms,
                &scale_pipeline.output_buffer,
                layer_beta,
            ),
            self.dimension,
            batch_size,
        );

        //Run bias pipeline
        bias_pipeline.run(encoder, self.dimension, batch_size);

        //Create vec for return
        let mut vec: Vec<wgpu::Buffer> = Vec::with_capacity(3);
        vec.push(normalization_pipeline.output_buffer);
        vec.push(batchnormprime_pipeline.output_buffer);
        
        //Update mutable values
        *data_mean = mean_update_pipeline.output_buffer;
        *data_var = var_update_pipeline.output_buffer;
        *batches_sampled = sample_update_pipeline.output_buffer;
        
        //Return
        (bias_pipeline.output_buffer, vec)
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
        let _layer_gamma = gpu_data.next().unwrap();
        let _layer_beta = gpu_data.next().unwrap();
        let _data_var =  gpu_data.next().unwrap();
        let _data_mean = gpu_data.next().unwrap();
        let _batches_sampled = gpu_data.next().unwrap();

        let mut gpu_data = backprop_data.into_iter();
        let layer_normed = gpu_data.next().unwrap();
        let layer_normprime = gpu_data.next().unwrap();
        let _layer_input = gpu_data.next().unwrap();

        //Create beta_grad pipeline
        let beta_grad_uniforms = {
            let uniform_data = [self.dimension as u32, batch_size as u32];
            device.create_buffer_init(
                &BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniform_data),
                    usage: wgpu::BufferUsage::UNIFORM,
                }
            )
        };

        let beta_grad_pipeline = pipelines::batchtotal::Pipeline::new::<f32>(anchor, (
                &beta_grad_uniforms,
                backprop_grad,
            ),
            self.dimension,
            batch_size,
        );

        //Run beta_grad pipeline
        beta_grad_pipeline.run(encoder, self.dimension, batch_size);

        //Create gamma_grad pipeline
        let gamma_grad_pipeline = pipelines::batchtotal::Pipeline::new::<f32>(anchor, (
                &beta_grad_uniforms,
                layer_normed,
            ),
            self.dimension,
            batch_size,
        );

        //Run gamma_grad pipeline
        gamma_grad_pipeline.run(encoder, self.dimension, batch_size);

        //Create input_grad pipeline
        let input_grad_pipeline = pipelines::elementmultiply::Pipeline::new::<f32>(anchor, (
                &beta_grad_uniforms,
                layer_normprime,
                backprop_grad,
            ),
            self.dimension,
            batch_size,
        );

        //Run input_grad pipeline
        input_grad_pipeline.run(encoder, self.dimension, batch_size);

        //Return
        let mut vec: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(5);
        vec.push(Some(gamma_grad_pipeline.output_buffer));
        vec.push(Some(beta_grad_pipeline.output_buffer));
        vec.push(None);
        vec.push(None);
        vec.push(None);
        (input_grad_pipeline.output_buffer, vec)
    }
}
