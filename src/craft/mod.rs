use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::Backend,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Tensor,
    },
};

pub mod utils;

#[derive(Module, Debug)]
struct Slice1<B: Backend> {
    feat0: Conv2d<B>,
    feat1: BatchNorm<B, 2>,
    // feat2: Relu
    feat3: Conv2d<B>,
    feat4: BatchNorm<B, 2>,
    // feat5: Relu
    feat6: MaxPool2d,
    feat7: Conv2d<B>,
    feat8: BatchNorm<B, 2>,
    // feat9: Relu
    feat10: Conv2d<B>,
    feat11: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
struct Slice2<B: Backend> {
    // feat12: Relu
    feat13: MaxPool2d,
    feat14: Conv2d<B>,
    feat15: BatchNorm<B, 2>,
    // feat16: Relu
    feat17: Conv2d<B>,
    feat18: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
struct Slice3<B: Backend> {
    // feat19: Relu
    feat20: Conv2d<B>,
    feat21: BatchNorm<B, 2>,
    // feat22: Relu
    feat23: MaxPool2d,
    feat24: Conv2d<B>,
    feat25: BatchNorm<B, 2>,
    // feat26: Relu
    feat27: Conv2d<B>,
    feat28: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
struct Slice4<B: Backend> {
    // feat29: Relu
    feat30: Conv2d<B>,
    feat31: BatchNorm<B, 2>,
    // feat32: Relu
    feat33: MaxPool2d,
    feat34: Conv2d<B>,
    feat35: BatchNorm<B, 2>,
    // feat36: Relu
    feat37: Conv2d<B>,
    feat38: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
struct Slice5<B: Backend> {
    max_pool: MaxPool2d,
    feat1: Conv2d<B>,
    feat2: Conv2d<B>,
}

impl<B: Backend> Slice5<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(x, self.max_pool, self.feat1, self.feat2)
    }
}

#[derive(Module, Debug)]
struct Vgg16Bn<B: Backend> {
    slice_1: Slice1<B>,
    slice_2: Slice2<B>,
    slice_3: Slice3<B>,
    slice_4: Slice4<B>,
    slice_5: Slice5<B>,
}

struct VggOutputs<B: Backend> {
    relu2_2: Tensor<B, 4>,
    relu3_2: Tensor<B, 4>,
    relu4_3: Tensor<B, 4>,
    relu5_3: Tensor<B, 4>,
    fc7: Tensor<B, 4>,
}

fn conv<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .init(device)
}

fn batch_norm<B: Backend>(out_c: usize, device: &B::Device) -> BatchNorm<B, 2> {
    BatchNormConfig::new(out_c).init(device)
}

fn max_pool() -> MaxPool2d {
    MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init()
}

impl<B: Backend> Slice1<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x,
            self.feat0,
            self.feat1,
            Relu,
            self.feat3,
            self.feat4,
            Relu,
            self.feat6,
            self.feat7,
            self.feat8,
            Relu,
            self.feat10,
            self.feat11,
            Relu
        )
    }
}

impl<B: Backend> Slice2<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x,
            self.feat13,
            self.feat14,
            self.feat15,
            Relu,
            self.feat17,
            self.feat18,
            Relu
        )
    }
}

impl<B: Backend> Slice3<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x,
            self.feat20,
            self.feat21,
            Relu,
            self.feat23,
            self.feat24,
            self.feat25,
            Relu,
            self.feat27,
            self.feat28,
            Relu
        )
    }
}

impl<B: Backend> Slice4<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x,
            self.feat30,
            self.feat31,
            Relu,
            self.feat33,
            self.feat34,
            self.feat35,
            Relu,
            self.feat37,
            self.feat38
        )
    }
}

impl<B: Backend> Vgg16Bn<B> {
    pub fn init(device: &B::Device) -> Self {
        let slice_1 = Slice1 {
            feat0: conv(3, 64, device),
            feat1: batch_norm(64, device),
            feat3: conv(64, 64, device),
            feat4: batch_norm(64, device),
            feat6: max_pool(),
            feat7: conv(64, 128, device),
            feat8: batch_norm(128, device),
            feat10: conv(128, 128, device),
            feat11: batch_norm(128, device),
        };
        let slice_2 = Slice2 {
            feat13: max_pool(),
            feat14: conv(128, 256, device),
            feat15: batch_norm(256, device),
            feat17: conv(256, 256, device),
            feat18: batch_norm(256, device),
        };
        let slice_3 = Slice3 {
            feat20: conv(256, 256, device),
            feat21: batch_norm(256, device),
            feat23: max_pool(),
            feat24: conv(256, 512, device),
            feat25: batch_norm(512, device),
            feat27: conv(512, 512, device),
            feat28: batch_norm(512, device),
        };
        let slice_4 = Slice4 {
            feat30: conv(512, 512, device),
            feat31: batch_norm(512, device),
            feat33: max_pool(),
            feat34: conv(512, 512, device),
            feat35: batch_norm(512, device),
            feat37: conv(512, 512, device),
            feat38: batch_norm(512, device),
        };

        let slice_5 = Slice5 {
            max_pool: MaxPool2dConfig::new([3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),
            feat1: Conv2dConfig::new([512, 1024], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(6, 6))
                .with_dilation([6, 6])
                .init(device),
            feat2: Conv2dConfig::new([1024, 1024], [1, 1]).init(device),
        };

        Self {
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> VggOutputs<B> {
        let x = self.slice_1.forward(x);
        let relu2_2 = x.clone();
        let x = self.slice_2.forward(x);
        let relu3_2 = x.clone();
        let x = self.slice_3.forward(x);
        let relu4_3 = x.clone();
        let x = self.slice_4.forward(x);
        let relu5_3 = x.clone();
        let fc7 = self.slice_5.forward(x);

        VggOutputs {
            relu2_2,
            relu3_2,
            relu4_3,
            relu5_3,
            fc7,
        }
    }
}

#[derive(Config, Debug)]
struct ConvBlockConfig {
    in_ch: usize,
    mid_ch: usize,
    out_ch: usize,
}

#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    batch_norm1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    batch_norm2: BatchNorm<B, 2>,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        let ConvBlockConfig {
            in_ch,
            mid_ch,
            out_ch,
        } = *self;
        ConvBlock {
            conv1: Conv2dConfig::new([in_ch + mid_ch, mid_ch], [1, 1]).init(device),
            batch_norm1: BatchNormConfig::new(mid_ch).init(device),
            conv2: Conv2dConfig::new([mid_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            batch_norm2: BatchNormConfig::new(out_ch).init(device),
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sequential!(
            x,
            self.conv1,
            self.batch_norm1,
            Relu,
            self.conv2,
            self.batch_norm2,
            Relu
        )
    }
}

#[derive(Module, Debug)]
pub struct Craft<B: Backend> {
    // Base network
    basenet: Vgg16Bn<B>,

    // U network
    upconv1: ConvBlock<B>,
    upconv2: ConvBlock<B>,
    upconv3: ConvBlock<B>,
    upconv4: ConvBlock<B>,

    conv_cls: Vec<Conv2d<B>>,
}

impl<B: Backend> Craft<B> {
    pub fn init(device: &B::Device) -> Self {
        let num_class = 2;
        Self {
            basenet: Vgg16Bn::init(device),

            upconv1: ConvBlockConfig::new(1024, 512, 256).init(device),
            upconv2: ConvBlockConfig::new(512, 256, 128).init(device),
            upconv3: ConvBlockConfig::new(256, 128, 64).init(device),
            upconv4: ConvBlockConfig::new(128, 64, 32).init(device),

            conv_cls: vec![
                Conv2dConfig::new([32, 32], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                Conv2dConfig::new([32, 32], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                Conv2dConfig::new([32, 16], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                Conv2dConfig::new([16, 16], [1, 1]).init(device),
                Conv2dConfig::new([16, num_class], [1, 1]).init(device),
            ],
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Base network
        let sources = self.basenet.forward(x);
        let shape_2 = sources.relu4_3.shape().dims;
        let shape_3 = sources.relu3_2.shape().dims;
        let shape_4 = sources.relu2_2.shape().dims;

        // U network
        let y = Tensor::cat(vec![sources.fc7, sources.relu5_3], 1);
        let y = self.upconv1.forward(y);

        let bilinear = InterpolateOptions::new(InterpolateMode::Bilinear);

        let y = interpolate(y, [shape_2[2], shape_2[3]], bilinear.clone());
        let y = Tensor::cat(vec![y, sources.relu4_3], 1);
        let y = self.upconv2.forward(y);

        let y = interpolate(y, [shape_3[2], shape_3[3]], bilinear.clone());
        let y = Tensor::cat(vec![y, sources.relu3_2], 1);
        let y = self.upconv3.forward(y);

        let y = interpolate(y, [shape_4[2], shape_4[3]], bilinear.clone());

        let y = Tensor::cat(vec![y, sources.relu2_2], 1);
        let feature = self.upconv4.forward(y);

        let mut y = feature.clone();

        for conv in self.conv_cls.iter().take(self.conv_cls.len() - 1) {
            y = conv.forward(y);
            y = Relu.forward(y);
        }

        let last = &self.conv_cls[self.conv_cls.len() - 1];
        let y = last.forward(y);

        (y.permute([0, 2, 3, 1]), feature)
    }
}
