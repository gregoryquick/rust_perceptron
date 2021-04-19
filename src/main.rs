mod pipelines;
mod data;

use futures::executor::block_on;

fn main() {
    const INPUT_DIM: usize = 28 * 28;
    const OUTPUT_DIM: usize = 10;
    let pipeline_anchor = block_on(pipelines::PipelineAnchor::new(INPUT_DIM, OUTPUT_DIM));
    let batch_size: usize = 4;
    let matrix_pipeline = pipelines::matrixdot::Pipeline::new::<f32>(&pipeline_anchor, batch_size);

    let data_set = data::DataSet::<data::mnist::Data>::new("train");
    let batch = data_set.generate_batch(batch_size);
    let labels = data::DataSet::<data::mnist::Data>::get_labels(&batch);
    println!("Batch labels:");
    println!("{:?}", labels);
    
}
