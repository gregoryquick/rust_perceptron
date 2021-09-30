mod layers;
mod cost;
pub mod perceptron;

pub enum LayerType {
    FullyConnected(usize),
    Batchnorm,
    Relu,
    Softmax,
}

pub enum CostFunction {
    SquaredError,
    CrossEntropy,
}
