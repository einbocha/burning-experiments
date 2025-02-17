pub mod training;

use burn::{
    nn::{
        loss::BinaryCrossEntropyLossConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig,
        Sigmoid,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::xor::XorBatch;

#[derive(Module, Debug)]
pub struct XorModel<B: Backend> {
    linear1: Linear<B>,
    leaky_relu2: LeakyRelu,
    linear3: Linear<B>,
    leaky_relu4: LeakyRelu,
    linear5: Linear<B>,
    sigmoid6: Sigmoid,
}

impl<B: Backend> XorModel<B> {
    pub fn forward(&self, ab_pairs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(ab_pairs);
        let x = self.leaky_relu2.forward(x);
        let x = self.linear3.forward(x);
        let x = self.leaky_relu4.forward(x);
        let x = self.linear5.forward(x);
        self.sigmoid6.forward(x)
    }
}

impl<B: Backend> XorModel<B> {
    pub fn forward_classification(
        &self,
        ab_pairs: Tensor<B, 2>,
        axorbs: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let prediction = self.forward(ab_pairs);
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&prediction.device())
            .forward(prediction.clone().reshape([-1]), axorbs.clone());

        ClassificationOutput::new(loss, prediction, axorbs)
    }
}

impl<B: AutodiffBackend> TrainStep<XorBatch<B>, ClassificationOutput<B>> for XorModel<B> {
    fn step(&self, batch: XorBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.ab_pairs, batch.axorbs);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<XorBatch<B>, ClassificationOutput<B>> for XorModel<B> {
    fn step(&self, batch: XorBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.ab_pairs, batch.axorbs)
    }
}

#[derive(Config, Debug)]
pub struct XorModelConfig {
    hidden_size: usize,
}

impl XorModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> XorModel<B> {
        XorModel {
            linear1: LinearConfig::new(2, 2).init(device),
            leaky_relu2: LeakyReluConfig::new().init(),
            linear3: LinearConfig::new(2, self.hidden_size).init(device),
            leaky_relu4: LeakyReluConfig::new().init(),
            linear5: LinearConfig::new(self.hidden_size, 1).init(device),
            sigmoid6: Sigmoid::new(),
        }
    }
}
