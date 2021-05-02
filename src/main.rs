mod pipelines;
mod data;

extern crate serde;

use rand::prelude::*;

use futures::executor::block_on;
//use std::thread;
use std::fs::File;

fn main() {
    const INPUT_DIM: usize = 28 * 28;
    const OUTPUT_DIM: usize = 10;
    let pipeline_anchor = block_on(pipelines::PipelineAnchor::new(INPUT_DIM, OUTPUT_DIM));
    const BATCH_SIZE: usize = 3;

    let data_set = data::DataSet::<data::mnist::Data>::new("train");
    let batch = data_set.generate_batch(BATCH_SIZE);
    let labels = data::DataSet::<data::mnist::Data>::get_labels(&batch);
    println!("Batch labels:");
    println!("{:?}", labels);

    //Actual computation bellow
    //Input data
    let batch_images = data::DataSet::<data::mnist::Data>::get_data(&batch);
    let input_data = {
        let mut vector: Vec<f32> = vec![0f32; INPUT_DIM * BATCH_SIZE];
        for (loc, data) in vector.iter_mut().zip(batch_images.into_iter()) {
            *loc = *data;
        }
        vector
    };    
    //Weights
    let mut rng = rand::thread_rng();
    use rand::distributions::Uniform;
    let dist = Uniform::new(-1.0,1.0);
    const GENERATE_NETWORK: bool = false;
    let network_weights = {
        let mut vector: Vec<f32> = vec![0f32; INPUT_DIM * OUTPUT_DIM];
        if GENERATE_NETWORK {
            for num in vector.iter_mut() {
                    *num = rng.sample(dist);
            }
            let file = File::create("weights/network.bin").unwrap();
            bincode::serialize_into(&file, &vector).unwrap();
            println!("Generated network weights!");
        }
        else {
            let file = File::open("weights/network.bin").unwrap();
            let weight_data: Vec<f32> = bincode::deserialize_from(&file).unwrap();
            vector = weight_data;
        }
        vector
    };
    
    //Compute predictions
    let prediction = block_on(pipelines::run_forward_pass::<f32>(&pipeline_anchor, &network_weights, &input_data, BATCH_SIZE)).unwrap();
    println!("Predictions:");
    println!("{:?}", prediction);

    //Compute error
    let label_data = {
        let mut vector: Vec<f32> = vec![0f32; OUTPUT_DIM * BATCH_SIZE];
        for (loc, data) in vector.iter_mut().zip(labels.into_iter()) {
            *loc = data;
        }
        vector
    };   
    let gradient = block_on(pipelines::run_backward_pass::<f32>(&pipeline_anchor, &network_weights, &input_data, &label_data, BATCH_SIZE)).unwrap();
    println!("Weight gradient:");
    println!("{:?}", gradient);

    
}
