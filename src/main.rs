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
    clippy::upper_case_acronyms,
    clippy::while_let_on_iterator,
    clippy::wildcard_imports,
    clippy::items_after_statements,
    clippy::clone_on_copy,
    non_upper_case_globals,
)]
#![allow(
    dead_code,
    unused_variables,
    incomplete_features,
)]

#![feature(generic_const_exprs)]

use std::error::Error;

use tracing::{span, event, Level};

mod device;
mod dispatch;
mod kernel;
mod tensor;

//use tracing::info;

/// The main function for the program
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    //Logging
    use std::env;
    env::set_var("RUST_BACKTRACE", "1");
    env::set_var("RUST_LOG", "info");

    tracing_subscriber::fmt::init();

    //Do stuffs
    let span = span!(Level::INFO, "Setup");
    let guard = span.enter();
    
    let pool = device::DevicePool::new().await?;
    event!(Level::INFO, "Got Devices");

    drop(guard);
    //Make tensor
    let span = span!(Level::INFO, "Tensors");
    let guard = span.enter();

    use crate::tensor::*;

    let gpu_0 = pool.gpus()[0];
    let gpu_1 = pool.gpus()[1];
    let cpu = pool.cpu();

    let tensor_0 = Tensor {
        device: cpu,
        tensor_layout: Strided {
            strides: [1],
        },
        shape: [2],
        data: TensorData::CPUStrided::<f32>(
            vec![
            1.0,
            0.0,
            ]
        ),
    };
    event!(Level::INFO, "Created a tensor");

    let tensor_a = tensor_0.into_device(gpu_0).await?;
    event!(Level::INFO, "Moved tensor to GPU");

    let tensor_b = tensor_a.clone().into_device(gpu_1).await?;
    event!(Level::INFO, "Cloned tensor");

    //let tensor_c = kernel::gpu::tensor_operations::product::strided::float32::forward(&tensor_a, &tensor_b);
    //event!(Level::INFO, "Multiplied tensors");

    let tensor_c = kernel::gpu::arithmetic_operations::elementwise_add::strided::float32::forward(gpu_0, &tensor_a, &tensor_b);
    event!(Level::INFO, "Added tensors");

    let tensor_1 = tensor_c.into_device(cpu).await?;
    event!(Level::INFO, "Moved tensor to CPU");
    
    use enum_extract::extract;
    let data = extract!(TensorData::CPUStrided(_), tensor_1.data).unwrap();
    println!("{:?}", data);
    event!(Level::INFO, "Printed tensor data");
    
    drop(guard);
    Ok(())
}


