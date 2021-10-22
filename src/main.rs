//! Project to run neural network on gpu using rust
//! GPU comunication is done using wgpu
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_sign_loss,
)]
#![allow(
    dead_code,
    unused_variables,
)]

use anyhow::{Result};

mod data;
mod device;
mod tensor;

use tensor::Tensor;
use device::Device;

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

    //Hook into devices
    let device_pool = device::DevicePool::new().await?;
    let gpu = &device_pool.devices[0];
    
    //Create Data
    let data = Tensor::CPUMatrix{
        data: vec![1.0,0.0,0.0,1.0],
        shape: (2, 2),
        stride: (2, 1),
    }.transfer_device(&Device::Cpu, gpu).await?;

    let data = data.transfer_device(gpu, &Device::Cpu).await?;

    match data {
        Tensor::CPUMatrix{data, ..} => {
            println!("{:?}", data);
        },
        Tensor::GPUMatrix{ .. } => {},
    };
    
    //Create channel
    let (tx, mut rx) = tokio::sync::broadcast::channel(16);
    #[allow(unreachable_code)]
    tokio::spawn(async move {
        loop {
            println!("received = {:?}", rx.recv().await?);
        }

        let result: Result<()> = Ok(());
        result
    });

    tx.send("Hello")?;
    tx.send("World!")?;


    Ok(())
}

/// Message for comunication between server and client
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum Message {
    TODO,
}
