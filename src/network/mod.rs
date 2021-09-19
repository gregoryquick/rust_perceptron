mod layers;
mod cost;
pub mod perceptron;

pub enum LayerType {
    DenseLayer(usize),
    Batchnorm,
}

pub enum CostFunction {
    SquaredError,
}
