mod xor;

use crate::xor::model::XorModelConfig;
use burn::backend::Wgpu;

type MyBackend = Wgpu<f32, i32>;

fn main() {
    let device = Default::default();
    let model = XorModelConfig::new(2).init::<MyBackend>(&device);

    println!("{}", model);
}
