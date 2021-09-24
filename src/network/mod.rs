mod layers;
mod cost;
pub mod perceptron;

pub enum LayerType {
    DenseLayer(usize),
    Batchnorm,
    Softmax(usize)
}

pub enum CostFunction {
    SquaredError,
    CrossEntropy,
}
