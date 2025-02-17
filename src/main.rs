mod data;
mod model;

use crate::model::xor::{
    training::{xor_training, XorTrainingConfig},
    XorModelConfig,
};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./artifacts";

    xor_training::<MyAutodiffBackend>(
        artifact_dir,
        XorTrainingConfig::new(XorModelConfig::new(3), AdamConfig::new()),
        device.clone(),
    );
}
