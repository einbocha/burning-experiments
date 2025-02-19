use crate::{
    data::xor::{xor_dataset, XorBatcher},
    model::xor::XorModelConfig,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

#[derive(Config)]
pub struct XorTrainingConfig {
    pub model: XorModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1000)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 432952)]
    pub seed: u64,
    #[config(default = 1.0e-2)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn xor_training<B: AutodiffBackend>(
    artifact_dir: &str,
    config: XorTrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = XorBatcher::<B>::new(device.clone());
    let batcher_valid = XorBatcher::<B::InnerBackend>::new(device.clone());

    let datasubset_train = xor_dataset();
    let datasubset_valid = xor_dataset();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(datasubset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(datasubset_valid);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
