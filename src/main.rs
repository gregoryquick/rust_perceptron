//! Project to run neural network on gpu using rust
//! GPU comunication is done using wgpu
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::match_wildcard_for_single_variants,
    clippy::trivially_copy_pass_by_ref,
)]
#![allow(
    dead_code,
    unused_variables,
)]

use anyhow::{Result};

mod autograd;
mod data;
mod device;

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
    
    //Create channel
    let (tx, mut rx) = tokio::sync::broadcast::channel(16);
    #[allow(unreachable_code)]
    tokio::spawn(async move {
        //Create manager for devices
        let device_pool = device::DevicePool::new().await?;
        let cpu = device_pool.cpu();
        let gpu = device_pool.gpu();

        //Tensor!
        let test_tensor = Tensor {
            device: cpu,
            interior_data: TensorData::CPUData{
                data: vec![1.0,0.0,0.0,1.0],
            },
            shape: (2, 2),
            stride: (2, 1),
        };
        
        let test_tensor = test_tensor.to(gpu).await?;

        //let test_tensor = test_tensor.to(cpu).await?;

        //Print some tensor info
        match &test_tensor.interior_data {
             TensorData::CPUData{data, ..} => {
                 println!("{:?}", data);
             },
             TensorData::GPUData{..} => {
                 println!("{}", test_tensor.size());
             },
        };

        //Graph things
        let dummy_connections = vec![(0, 2), (1, 2), (1, 0), (2, 3), (3, 4)];
        let graph = autograd::ComputeGraph::dummy_new(gpu, dummy_connections);

        graph.dummy_test();
        
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

/// Message for comunication between client and processor
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum Message {
    TODO,
}
