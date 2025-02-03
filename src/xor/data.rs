use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    prelude::*,
};

struct XorItem {
    ab: [bool; 2],
    axorb: bool,
}

fn xor_dataset() -> InMemDataset<XorItem> {
    let mut pairs: Vec<XorItem> = vec![];

    for a in 0..=1 {
        for b in 0..=1 {
            pairs.push(XorItem {
                ab: [a == 1, b == 1],
                axorb: a != b,
            })
        }
    }

    InMemDataset::new(pairs)
}

pub struct XorBatch<B: Backend> {
    ab_pairs: Tensor<B, 2>,
    axorbs: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct XorBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> XorBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<XorItem, XorBatch<B>> for XorBatcher<B> {
    fn batch(&self, items: Vec<XorItem>) -> XorBatch<B> {
        let ab_pairs = items
            .iter()
            .map(|item| TensorData::from(item.ab))
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 2]))
            .collect();

        let axorbs = items
            .iter()
            .map(|item| TensorData::from([item.axorb]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .collect();

        let ab_pairs = Tensor::cat(ab_pairs, 0).to_device(&self.device);
        let axorbs = Tensor::cat(axorbs, 0).to_device(&self.device);

        XorBatch { ab_pairs, axorbs }
    }
}
