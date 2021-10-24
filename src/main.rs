//! Project to run neural network on gpu using rust
//! GPU comunication is done using wgpu
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::match_wildcard_for_single_variants,
)]
#![allow(
    dead_code,
    unused_variables,
)]

use anyhow::{Result};

mod data;
mod device;

use device::tensor::Tensor;

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
        let data = Tensor::CPUMatrix{
            device: gpu,
            data: vec![1.0,0.0,0.0,1.0],
            shape: (2, 2),
            stride: (2, 1),
        };
        
        let data = data.to(gpu).await?;

        let data = data.to(cpu).await?;

        //Print some tensor info
        match data {
             Tensor::CPUMatrix{data, ..} => {
                 println!("{:?}", data);
             },
             Tensor::GPUMatrix{..} => {
                 println!("{}", data.size());
             },
        };

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

/// Message for comunication between client and proccesor
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
enum Message {
    TODO,
}
