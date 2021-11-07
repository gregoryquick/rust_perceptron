//! Project to run neural network on gpu using rust
//! GPU comunication is done using wgpu
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
            println!("received = {:?}", rx.recv().await?);
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
    let dummy_data = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
                1.0,0.0,
                0.0,1.0,
                1.0,0.0,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };
        
    let dummy_data = dummy_data.to(gpu).await?;

    let dummy_matrix = Tensor {
        device: cpu,
        interior_data: TensorData::CPUData{
            data: vec![
            0.0,1.0,
            1.0,0.0,
            0.0,1.0,
            ],
        },
        shape: (3, 2),
        stride: (2, 1),
    };
        
    let dummy_matrix = dummy_matrix.to(gpu).await?;

    //Graph things
    let mut graph = autograd::graph::ComputeGraph::new(gpu);

    let data_tensor = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(dummy_data, None)));
        
    let matrix_tensor = graph.add_operation(Box::new(autograd::tensor::GraphTensor::new(dummy_matrix, None)));
        
    let addition = graph.add_operation(Box::new(autograd::matrix_operations::matrixadd::MatrixAdd::new()));
        
    //let output_probe = graph.add_operation(Box::new(autograd::output::OutputProbe::new()));
        
    graph.print_graph();

    println!("{:?}", graph.binds());

    //graph.link(addition, output_probe, (0, 0))?;

    graph.link(data_tensor, addition, (0, 0))?;

    graph.link(matrix_tensor, addition, (0, 1))?;

    graph.print_graph();

    println!("{:?}", graph.binds());

    graph.test()?;

    let outputs_to_read = vec![(addition, 0)];

    let output = graph.get_outputs(&outputs_to_read)?;

    for tensor in output {
        let cpu_tensor = tensor.to(cpu).await?;
        match &cpu_tensor.interior_data {
            TensorData::CPUData{data, ..} => {
                println!("{:?}", data);
            },
            TensorData::GPUData{..} => {
                println!("{}", cpu_tensor.size());
            },
        };
    }

    Ok(())
}

/// Message for comunication between client and processor
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum Message {
    TODO,
}
