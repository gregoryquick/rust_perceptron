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
)]
#![allow(
    dead_code,
    unused_variables,
)]

use std::error::Error;

use tracing::{span, event, Level};

mod device;
mod dispatch;
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

    let gpu = pool.gpus()[0];
    let cpu = pool.cpu();

    let tensor_0 = tensor::Tensor {
        device: cpu,
        tensor_layout: tensor::Strided {
            strides: [5, 1],
        },
        shape: [5, 5],
        data: tensor::TensorData::CPUStrided::<f32>(
            vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
            ]
        ),
    };
    event!(Level::INFO, "Created a tensor");

    let tensor_a = tensor_0.to_device(gpu).await?;
    event!(Level::INFO, "Moved tensor to GPU");

    let tensor_b = tensor_a.clone();
    event!(Level::INFO, "Cloned GPU tensor");

    drop(guard);
    Ok(())
}


