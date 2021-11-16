//! Project to run neural network on gpu using rust
//! GPU comunication is done using wgpu
#![feature(slice_group_by)]
#![feature(drain_filter)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::match_wildcard_for_single_variants,
    clippy::trivially_copy_pass_by_ref,
    clippy::into_iter_on_ref,
    clippy::too_many_lines,
)]
#![allow(
    dead_code,
    unused_variables,
)]

use anyhow::{Result};

mod autograd;
mod data;
mod device;
mod gpu;


use device::tensor::{Tensor, TensorData};

/// The main function for the program
#[tokio::main]
async fn main() -> Result<()> {
    //Logging
    use std::env;
    env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();

    //Global vars
    let output_size: usize = 10;
    let batch_size: usize = 64;
    
    //Load data
    let training_data = data::mnist::load_data("train")?;
    let test_data = data::mnist::load_data("t10k")?;
    
    graph_stuff().await?;

    //Create channel
    let (tx, mut rx) = tokio::sync::broadcast::channel(16);
    #[allow(unreachable_code)]
    tokio::spawn(async move {

        //graph_stuff().await?;
        
        //Recive instructions
        loop {
            rx.recv().await?;
            //println!("received = {:?}", rx.recv().await?);
        }
        
        //Help compiler know return type is anyhow::Result
        let result: Result<()> = Ok(());
        result
    });

    tx.send("Hello")?;
    tx.send("World!")?;

    Ok(())
}

#[allow(clippy::missing_errors_doc)]
pub async fn graph_stuff() -> Result<()> {
    //Create manager for devices
    let device_pool = device::DevicePool::new().await?;
    let cpu = device_pool.cpu();
    let gpu = device_pool.gpu();

    //Tensors!
    //Tensor 0
    let tensor_0 = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
                0.0,0.1,
                0.1,0.1,
                0.0,0.1,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };

    print_tensor(&tensor_0);
        
    let tensor_0 = tensor_0.to(gpu).await?;

    //Tensor 1
    let tensor_1 = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
            0.0,0.6,
            0.6,0.0,
            0.0,0.0,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };

    print_tensor(&tensor_1);
        
    let tensor_1 = tensor_1.to(gpu).await?;

    //Tensor 2
    let tensor_2 = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
            0.1,0.1,
            0.1,0.0,
            0.1,0.0,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };

    print_tensor(&tensor_2);
        
    let tensor_2 = tensor_2.to(gpu).await?;

    //Label tensor
    let tensor_label = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
            0.1,0.8,
            0.1,0.1,
            0.8,0.1,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };
    
    print_tensor(&tensor_label);
    
    let tensor_label = tensor_label.to(gpu).await?;

    //Graph things
    let mut graph = autograd::graph::ComputeGraph::new(gpu);

    let graph_tensor_0 = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(tensor_0, None)));
        
    let graph_tensor_1 = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(tensor_1, None)));

    let graph_tensor_2 = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(tensor_2, None)));

    let graph_tensor_3 = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(tensor_label, None)));
        
    let addition_0 = graph.add_operation(Box::new(autograd::matrix_operations::matrixadd::MatrixAdd::new()));

    let addition_1 = graph.add_operation(Box::new(autograd::matrix_operations::matrixadd::MatrixAdd::new()));

    let cross_entropy = graph.add_operation(Box::new(autograd::cost_functions::crossentropy::CrossEntropy::new()));
    
    graph.link((graph_tensor_0, 0), (addition_0, 0))?;

    graph.link((graph_tensor_1, 0), (addition_0, 1))?;

    graph.link((addition_0, 0), (addition_1, 0))?;

    graph.link((graph_tensor_2, 0), (addition_1, 1))?;

    graph.link((addition_1, 0), (cross_entropy, 0))?;

    graph.link((graph_tensor_3, 0), (cross_entropy, 1))?;

    graph.print_graph();
    
    println!("Bindings: {:?}", graph.binds());

    //Run graph
    graph.forward()?;

    let outputs_to_read = vec![(addition_1, 0), (cross_entropy, 0)];

    let output = graph.get_outputs(&outputs_to_read)?;

    for tensor in output {
        let cpu_tensor = tensor.to(cpu).await?;
        print_tensor(&cpu_tensor);
    }

    let start_backprop_with = vec![cross_entropy];

    graph.backward(&start_backprop_with)?;

    Ok(())
}

fn print_tensor(tensor: &Tensor) {
    match &tensor.interior_data {
        TensorData::CPUData{data, ..} => {
            println!("{:?}", data);
        },
        TensorData::GPUData{..} => {
            println!("{}", tensor.size());
        },
    };
}

/// Message for comunication between client and processor
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum Message {
    TODO,
}
