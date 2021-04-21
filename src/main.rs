mod pipelines;
mod data;

extern crate serde;

use rand::prelude::*;

use futures::executor::block_on;
use std::thread;
use std::fs::File;

fn main() {
    const INPUT_DIM: usize = 28 * 28;
    const OUTPUT_DIM: usize = 10;
    let pipeline_anchor = block_on(pipelines::PipelineAnchor::new(INPUT_DIM, OUTPUT_DIM));
    const BATCH_SIZE: usize = 1;
    //let matrix_pipeline = pipelines::matrixdot::Pipeline::new::<f32>(&pipeline_anchor, (None, None, None), batch_size);
    //let activation_pipeline = pipelines::leakyrelu::Pipeline::new::<f32>(&pipeline_anchor, (Some(matrix_pipeline.output_buffer), None), batch_size);

    let data_set = data::DataSet::<data::mnist::Data>::new("train");
    let batch = data_set.generate_batch(BATCH_SIZE);
    let labels = data::DataSet::<data::mnist::Data>::get_labels(&batch);
    println!("Batch labels:");
    println!("{:?}", labels);

    //Actual computation bellow
    //Format input data
    let batch_images = data::DataSet::<data::mnist::Data>::get_data(&batch);
    let input_data = {
        const DATA_SIZE: usize = INPUT_DIM * BATCH_SIZE;
        let mut vector: [f32; DATA_SIZE] = [0f32; DATA_SIZE];
        for (loc, data) in vector.iter_mut().zip(batch_images.into_iter().flatten()) {
            *loc = *data;
        }
        vector
    };

    //Weights
    const WEIGHT_SIZE: usize = INPUT_DIM * OUTPUT_DIM;
    let mut rng = rand::thread_rng();
    use rand::distributions::Uniform;
    let dist = Uniform::new(-1.0,1.0);
    let network_weights = {
        let mut vector: [f32; WEIGHT_SIZE] = [0f32; WEIGHT_SIZE];
        for num in vector.iter_mut() {
                *num = rng.sample(dist);
        }
        //let file = File::create("weights/network.bin").unwrap();
        //bincode::serialize_into(&file, &Weights::<WEIGHT_SIZE>{arr: vector}).unwrap();
        vector
    };
    
    //Compute
    let prediction = block_on(pipelines::run_forward_pass::<f32>(&pipeline_anchor, &network_weights, &input_data, BATCH_SIZE)).unwrap();
    println!("Predictions:");
    println!("{:?}", prediction);

    
}
