use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
struct LastConv<B: Backend> {
    feat0: Conv2d<B>,
    feat1: BatchNorm<B, 2>,
    // feat2: Relu
    feat3: Conv2d<B>,
    feat4: BatchNorm<B, 2>,
    // feat5: Relu
    feat6: Conv2d<B>,
    feat7: BatchNorm<B, 2>,
    // feat8: Relu
}

impl<B: Backend> LastConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x, self.feat0, self.feat1, Relu, self.feat3, self.feat4, Relu, self.feat6, self.feat7,
            Relu
        )
    }
}

#[derive(Config, Debug)]
struct AsppConfig {
    dilation: usize,
}

#[derive(Module, Debug)]
struct Aspp<B: Backend> {
    feat0: Conv2d<B>,
    feat1: BatchNorm<B, 2>,
    // feat2: Relu
    feat3: Conv2d<B>,
    feat4: BatchNorm<B, 2>,
    // feat5: Relu
    feat6: Conv2d<B>,
}

impl AsppConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Aspp<B> {
        let d = self.dilation;
        Aspp {
            feat0: Conv2dConfig::new([64, 128], [3, 3])
                .with_dilation([d, d])
                .with_padding(PaddingConfig2d::Explicit(d, d))
                .init(device),
            feat1: BatchNormConfig::new(127).init(device),
            feat3: Conv2dConfig::new([128, 128], [1, 1]).init(device),
            feat4: BatchNormConfig::new(128).init(device),
            feat6: Conv2dConfig::new([128, 1], [1, 1]).init(device),
        }
    }
}

impl<B: Backend> Aspp<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(x, self.feat0, self.feat1, Relu, self.feat3, self.feat4, Relu, self.feat6)
    }
}

#[derive(Module, Debug)]
pub struct RefineNet<B: Backend> {
    last_conv: LastConv<B>,

    aspp1: Aspp<B>,
    aspp2: Aspp<B>,
    aspp3: Aspp<B>,
    aspp4: Aspp<B>,
}

impl<B: Backend> RefineNet<B> {
    pub fn init(device: &B::Device) -> Self {
        let last_conv = LastConv {
            feat0: Conv2dConfig::new([34, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            feat1: BatchNormConfig::new(64).init(device),
            feat3: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            feat4: BatchNormConfig::new(64).init(device),
            feat6: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            feat7: BatchNormConfig::new(64).init(device),
        };

        Self {
            last_conv,
            aspp1: AsppConfig::new(6).init(device),
            aspp2: AsppConfig::new(12).init(device),
            aspp3: AsppConfig::new(18).init(device),
            aspp4: AsppConfig::new(24).init(device),
        }
    }

    pub fn forward(&self, y: Tensor<B, 4>, upconv4: Tensor<B, 4>) -> Tensor<B, 4> {
        let refine = Tensor::cat(vec![y.permute([0, 3, 1, 2]), upconv4], 1);
        let refine = self.last_conv.forward(refine);

        let aspp1 = self.aspp1.forward(refine.clone());
        let aspp2 = self.aspp2.forward(refine.clone());
        let aspp3 = self.aspp3.forward(refine.clone());
        let aspp4 = self.aspp4.forward(refine);

        let out = aspp1 + aspp2 + aspp3 + aspp4;
        out.permute([0, 2, 3, 1])
    }
}
