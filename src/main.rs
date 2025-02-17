mod data;
mod model;

use crate::model::xor::XorModelConfig;
use burn::backend::Wgpu;

type MyBackend = Wgpu<f32, i32>;

// !!!!!!!!!!!!!!!!!!!!         Todo: impl tests for the data module            !!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!         Todo: Go on after "Let us move on to establishing the practical training configuration." in the burn book under Training.            !!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!         Todo: Dont forget to put the next code block into a train module/file (the module could also be a sub-module of the model module)            !!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!         Todo: tests are also overdue for the model module             !!!!!!!!!!!!!!!!!!!!!

fn main() {
    let device = Default::default();
    let model = XorModelConfig::new(2).init::<MyBackend>(&device);

    println!("{}", model);
}
