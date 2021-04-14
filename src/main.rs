mod pipelines;
mod data;

use futures::executor::block_on;

fn main() {
    const INPUT_DIM: usize = 28 * 28;
    const OUTPUT_DIM: usize = 10;
    let pipeline_anchor = block_on(pipelines::PipelineAnchor::new(INPUT_DIM, OUTPUT_DIM));
    let batch_size: usize = 2;
    let matrix_pipeline = pipelines::matrixdot::Pipeline::new::<f32>(&pipeline_anchor, batch_size);
}
