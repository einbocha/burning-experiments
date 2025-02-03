use burn::{
    nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Sigmoid},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct XorModel<B: Backend> {
    linear1: Linear<B>,
    leaky_relu2: LeakyRelu,
    linear3: Linear<B>,
    sigmoid4: Sigmoid,
}

#[derive(Config, Debug)]
pub struct XorModelConfig {
    hidden_size: usize,
}

impl XorModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> XorModel<B> {
        XorModel {
            linear1: LinearConfig::new(2, self.hidden_size).init(device),
            leaky_relu2: LeakyReluConfig::new().init(),
            linear3: LinearConfig::new(self.hidden_size, 1).init(device),
            sigmoid4: Sigmoid::new(),
        }
    }
}

impl<B: Backend> XorModel<B> {
    pub fn forward(&self, ab_pairs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = ab_pairs;
        let x = self.linear1.forward(x);
        let x = self.leaky_relu2.forward(x);
        let x = self.linear3.forward(x);
        self.sigmoid4.forward(x)
    }
}
