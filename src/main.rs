mod pipelines;
mod data;

extern crate serde;

//use rand::prelude::*;

use futures::executor::block_on;
//use std::thread;
use std::fs::File;

fn main() {
    gradient_descent::<f32>("weights/network.bin", 24, 0.1f32, 12);
}

fn gradient_descent<T: bytemuck::Pod + serde::Serialize>(weight_read_location: &str, number_of_runs: usize, learning_rate: T, batch_size: usize) {
    let input_dim: usize = 28 * 28;
    let output_dim: usize = 10;
    let validation_batch_size: usize = 2 * batch_size;

    let pipeline_anchor = block_on(pipelines::PipelineAnchor::new(input_dim, output_dim));
    let data_set = data::DataSet::<data::mnist::Data>::new("train");
    let validation_set = data::DataSet::<data::mnist::Data>::new("t10k");

    let type_size = std::mem::size_of::<T>();
    let device = &pipeline_anchor.device;

    //Load weights
    use wgpu::util::{BufferInitDescriptor, DeviceExt};

    let network_weights = {
        let file = File::open(weight_read_location).unwrap();
        let weight_data: Vec<f32> = bincode::deserialize_from(&file).unwrap();
        weight_data
    };

    let mut weight_data_buffer: wgpu::Buffer = device.create_buffer_init(
        &BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&network_weights[..]),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        }
    );

    //Create staging buffer for loading out of gpu
    let staging_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (type_size * &pipeline_anchor.input_size * &pipeline_anchor.output_size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        }
    );

    //Per batch operations
    for batch_index in 0..number_of_runs {
        let batch = data_set.generate_batch(batch_size);
        let batch_labels = data::DataSet::<data::mnist::Data>::get_labels(&batch);
        let batch_images = data::DataSet::<data::mnist::Data>::get_data(&batch);
        //Create command encoder
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None 
            }
        );

        //Load data for pipelines
        let input_data = {
            let mut vector: Vec<f32> = vec![0f32; &pipeline_anchor.input_size * batch_size];
            for (loc, data) in vector.iter_mut().zip(batch_images.into_iter()) {
                *loc = *data;
            }
            vector
        };
        let input_data_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input_data[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        let label_data_buffer = device.create_buffer_init(
            &BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&batch_labels[..]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            }
        );

        
        //Create weight application pipeline
        let weight_pipeline = pipelines::applyweights::Pipeline::new::<T>(&pipeline_anchor, (
            Some(weight_data_buffer),
            Some(input_data_buffer),
            None,
        ), batch_size);

        //Run weight pipeline
        weight_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);

        //Create activation pipeline
        let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<T>(&pipeline_anchor, (
            Some(weight_pipeline.output_buffer),
            None,
        ), batch_size);

        //Run activation pipeline
        activation_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);

        //Create loss calculation pipeline
        let loss_pipeline = pipelines::loss::Pipeline::new::<T>(&pipeline_anchor, (
            Some(activation_pipeline.output_buffer),
            Some(label_data_buffer),
            None,
        ), batch_size);

        //Run loss pipeline
        loss_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);

        //Create prediction sensitivity pipeline
        let sensitivity_pipeline = pipelines::leakyreluprime::Pipeline::new::<T>(&pipeline_anchor, (
            Some(activation_pipeline.input_buffer),
            None
        ), batch_size);

        //Run sensitivity pipeline
        sensitivity_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);
        
        //Create error backpropagation pipeline
        let backprop_pipeline = pipelines::backprop::Pipeline::new::<T>(&pipeline_anchor, (
            Some(loss_pipeline.output_buffer),
            Some(sensitivity_pipeline.output_buffer),
            Some(weight_pipeline.input_buffer),
            None,
        ), batch_size);

        //Run backprop pipeline
        backprop_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);

        //Create gradient descent pipeline
        let descendgrad_pipeline = pipelines::descendgrad::Pipeline::new::<T>(&pipeline_anchor, (
            Some(weight_pipeline.weight_buffer),
            Some(backprop_pipeline.output_buffer),
            None,
        ), learning_rate, batch_size);

        //Run gradient descent step
        descendgrad_pipeline.run(&pipeline_anchor, &mut encoder, batch_size);

        //Write out of gpu if on last iteration
        if (batch_index + 1) == number_of_runs {
            encoder.copy_buffer_to_buffer(
                &descendgrad_pipeline.output_buffer, 0,
                &staging_buffer, 0,
                (type_size * &pipeline_anchor.input_size * &pipeline_anchor.output_size) as wgpu::BufferAddress,
            );
        }

        //Submit commands to gpu que
        let queue = &pipeline_anchor.queue;
        queue.submit(Some(encoder.finish()));

        //Set up buffers for next iteration
        weight_data_buffer = descendgrad_pipeline.input_buffer_a;
        if false {
            println!("{}", batch_index);
        }
        else {
            weight_data_buffer = get_error_and_return_weights(&pipeline_anchor, &validation_set, weight_data_buffer, validation_batch_size);
        }
    }

    //Create future of the computation
    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
    
    //Wait for computation to complete
    device.poll(wgpu::Maintain::Wait);

    block_on(async {
        match buffer_future.await {
            Ok(()) => {
                //Get buffer contents
                let data = buffer_slice.get_mapped_range();
                //Convert to T and apply activation function
                let result: Vec<T> = data.chunks_exact(type_size).map(|b| *bytemuck::from_bytes::<T>(b)).collect();
                
                //Drop mapped view
                drop(data);
                //Unmap buffer
                staging_buffer.unmap();

                let file = File::create(weight_read_location).unwrap();
                bincode::serialize_into(&file, &result).unwrap();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        } 
    });
}

pub fn get_error_and_return_weights(
    anchor: &pipelines::PipelineAnchor,
    data_set: &data::DataSet<data::mnist::Data>,
    network_weights: wgpu::Buffer,
    validation_batch_size: usize,
    ) -> wgpu::Buffer {
    let batch = data_set.generate_batch(validation_batch_size);
    let batch_labels = data::DataSet::<data::mnist::Data>::get_labels(&batch);
    let batch_images = data::DataSet::<data::mnist::Data>::get_data(&batch);

    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; anchor.input_size * validation_batch_size];
        for (loc, data) in vector.iter_mut().zip(batch_images.into_iter()) {
            *loc = *data;
        }
        vector
    };

    let (error, weights) = block_on(pipelines::run_batch_error::<f32>(anchor,network_weights, &input_data, &batch_labels, validation_batch_size)).unwrap();

    println!("{}", error);

    weights
}

