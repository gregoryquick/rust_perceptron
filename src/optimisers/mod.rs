use crate::pipelines;
use crate::network;

pub struct Stochasticgradientdescent {
    network: network::NeuralNetwork,
    learning_rate: f32,
}

impl Stochasticgradientdescent {
    pub fn new(network: network::NeuralNetwork, learning_rate: f32) -> Self{
        Stochasticgradientdescent {
            network,
            learning_rate,
        }
    }
}
