mod data;
mod model;

use crate::{
    data::xor::XorBatcher,
    model::xor::{
        training::{xor_training, XorTrainingConfig},
        XorModelConfig,
    },
};
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
};
use data::xor::xor_dataset;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./artifacts";

    let model_config = XorModelConfig::new(3);
    let mut model = model_config.init::<MyBackend>(&device);

    xor_training::<MyAutodiffBackend>(
        artifact_dir,
        XorTrainingConfig::new(model_config, AdamConfig::new()),
        device.clone(),
    );

    model = model
        .load_file(
            format!("{}/model", artifact_dir),
            &CompactRecorder::new(),
            &device,
        )
        .expect("Should be able to load the model weights from the provided file");

    let batcher = XorBatcher::<MyBackend>::new(device.clone());

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(4)
        .build(xor_dataset());

    for item in dataloader.iter() {
        let classification = model.forward(item.ab_pairs);
        println!("target: {} prediction: {}", item.axorbs, classification);
    }
}
