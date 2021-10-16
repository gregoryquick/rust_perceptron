mod layers;
mod cost;
pub mod perceptron;

#[allow(dead_code)]
pub enum LayerType {
    FullyConnected(usize),
    Batchnorm,
    Relu,
    Softmax,
}

#[allow(dead_code)]
pub enum CostFunction {
    SquaredError,
    CrossEntropy,
}
