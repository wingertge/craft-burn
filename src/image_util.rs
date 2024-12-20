use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Device, ElementConversion, Shape, Tensor, TensorData,
    },
};
use image::{DynamicImage, Rgb, RgbImage};

#[derive(Config, Debug)]
pub struct NormalizeMeanVarianceConfig {
    #[config(default = "[0.485, 0.456, 0.406]")]
    mean: [f64; 3],
    #[config(default = "[0.229, 0.224, 0.225]")]
    variance: [f64; 3],
}

impl NormalizeMeanVarianceConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NormalizeMeanVariance<B> {
        let mean = TensorData::new(self.mean.to_vec(), Shape::new([1, 3, 1, 1]));
        let mean = Tensor::from_data(mean, device);
        let variance = TensorData::new(self.variance.to_vec(), Shape::new([1, 3, 1, 1]));
        let variance = Tensor::from_data(variance, device);

        NormalizeMeanVariance { mean, variance }
    }
}

#[derive(Module, Debug)]
pub struct NormalizeMeanVariance<B: Backend> {
    mean: Tensor<B, 4>,
    variance: Tensor<B, 4>,
}

impl<B: Backend> NormalizeMeanVariance<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = x - self.mean.clone();
        x / self.variance.clone()
    }
}

pub struct ResizeResult<B: Backend> {
    pub image: Tensor<B, 4>,
    pub ratio: f32,
    pub size_heatmap: (usize, usize),
}

pub fn resize_aspect_ratio<B: Backend>(
    img: DynamicImage,
    square_size: usize,
    mag_ratio: f32,
    device: &Device<B>,
) -> ResizeResult<B> {
    let start = Instant::now();

    let height = img.height();
    let width = img.width();

    let image_f32 = TensorData::new::<f32, _>(
        img.to_rgb32f().into_vec(),
        Shape::new([1, height as usize, width as usize, 3]),
    );
    let tensor = Tensor::from_data(image_f32, device).permute([0, 3, 1, 2]);

    let max = height.max(width) as f32;

    let mut target_size = mag_ratio * max;
    if target_size > square_size as f32 {
        target_size = square_size as f32
    }

    let ratio = target_size / max;

    let target_h = (height as f32 * ratio) as usize;
    let target_w = (width as f32 * ratio) as usize;

    let image = interpolate(
        tensor,
        [target_h, target_w],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    let target_h_32 = target_h.div_ceil(32) * 32;
    let target_w_32 = target_w.div_ceil(32) * 32;

    let padded = image.pad(
        (0, target_w_32 - target_w, 0, target_h_32 - target_h),
        0.0.elem(),
    );

    let size_heatmap = (target_h_32 / 2, target_w_32 / 2);

    println!("Resize took {:?}", Instant::now() - start);

    ResizeResult {
        image: padded,
        ratio,
        size_heatmap,
    }
}

pub fn float_to_color_map<B: Backend>(image: Tensor<B, 4>) -> DynamicImage {
    let [_, height, width, _] = image.shape().dims();
    let image = image.clamp(0.0, 1.0) * 255.0;
    let image_data = image.into_data().convert::<u8>().to_vec::<u8>().unwrap();
    let buf = RgbImage::from_par_fn(width as u32, height as u32, |x, y| {
        let value = image_data[y as usize * width + x as usize];
        let r = color_map::jet::R[value as usize] * 255.0;
        let g = color_map::jet::G[value as usize] * 255.0;
        let b = color_map::jet::B[value as usize] * 255.0;
        Rgb([r as u8, g as u8, b as u8])
    });
    buf.into()
}

pub mod color_map {
    pub mod jet {
        pub const R: [f32; 256] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.005_882_353,
            0.021_568_628,
            0.037_254_903,
            0.052_941_177,
            0.068_627_454,
            0.084_313_73,
            0.1,
            0.115_686_275,
            0.131_372_56,
            0.147_058_83,
            0.162_745_1,
            0.178_431_38,
            0.194_117_65,
            0.209_803_92,
            0.225_490_2,
            0.241_176_47,
            0.256_862_76,
            0.272_549_03,
            0.288_235_3,
            0.303_921_58,
            0.319_607_85,
            0.335_294_13,
            0.350_980_4,
            0.366_666_67,
            0.382_352_95,
            0.398_039_22,
            0.413_725_5,
            0.429_411_77,
            0.445_098_04,
            0.460_784_32,
            0.476_470_6,
            0.492_156_86,
            0.507_843_14,
            0.523_529_4,
            0.539_215_7,
            0.554_901_96,
            0.570_588_23,
            0.586_274_5,
            0.601_960_8,
            0.617_647_05,
            0.633_333_3,
            0.649_019_6,
            0.664_705_9,
            0.680_392_15,
            0.696_078_4,
            0.711_764_7,
            0.727_450_97,
            0.743_137_24,
            0.758_823_5,
            0.774_509_8,
            0.790_196_06,
            0.805_882_33,
            0.821_568_6,
            0.837_254_9,
            0.852_941_16,
            0.868_627_4,
            0.884_313_7,
            0.9,
            0.915_686_25,
            0.931_372_5,
            0.947_058_8,
            0.962_745_1,
            0.978_431_34,
            0.994_117_6,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.986_274_5,
            0.970_588_2,
            0.954_901_93,
            0.939_215_66,
            0.923_529_4,
            0.907_843_1,
            0.892_156_84,
            0.876_470_57,
            0.860_784_3,
            0.845_098,
            0.829_411_75,
            0.813_725_5,
            0.798_039_2,
            0.782_352_9,
            0.766_666_65,
            0.750_980_4,
            0.735_294_1,
            0.719_607_83,
            0.703_921_56,
            0.688_235_3,
            0.672_549,
            0.656_862_74,
            0.641_176_46,
            0.625_490_2,
            0.609_803_9,
            0.594_117_64,
            0.578_431_37,
            0.562_745_1,
            0.547_058_8,
            0.531_372_55,
            0.515_686_3,
            0.5,
        ];
        pub const G: [f32; 256] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001_960_784_4,
            0.017_647_06,
            0.033_333_335,
            0.049_019_61,
            0.064_705_886,
            0.080_392_16,
            0.096_078_43,
            0.111_764_71,
            0.127_450_99,
            0.143_137_26,
            0.158_823_53,
            0.174_509_81,
            0.190_196_08,
            0.205_882_36,
            0.221_568_63,
            0.237_254_9,
            0.252_941_2,
            0.268_627_46,
            0.284_313_74,
            0.3,
            0.315_686_3,
            0.331_372_56,
            0.347_058_83,
            0.362_745_1,
            0.378_431_38,
            0.394_117_65,
            0.409_803_93,
            0.425_490_2,
            0.441_176_47,
            0.456_862_75,
            0.472_549_02,
            0.488_235_3,
            0.503_921_57,
            0.519_607_84,
            0.535_294_1,
            0.550_980_4,
            0.566_666_66,
            0.582_352_94,
            0.598_039_2,
            0.613_725_5,
            0.629_411_76,
            0.645_098_03,
            0.660_784_3,
            0.676_470_6,
            0.692_156_85,
            0.707_843_1,
            0.723_529_4,
            0.739_215_7,
            0.754_901_95,
            0.770_588_2,
            0.786_274_5,
            0.801_960_77,
            0.817_647_04,
            0.833_333_3,
            0.849_019_6,
            0.864_705_86,
            0.880_392_13,
            0.896_078_4,
            0.911_764_7,
            0.927_450_95,
            0.943_137_2,
            0.958_823_5,
            0.974_509_8,
            0.990_196_05,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.990_196_05,
            0.974_509_8,
            0.958_823_5,
            0.943_137_2,
            0.927_450_95,
            0.911_764_7,
            0.896_078_4,
            0.880_392_13,
            0.864_705_86,
            0.849_019_6,
            0.833_333_3,
            0.817_647_04,
            0.801_960_77,
            0.786_274_5,
            0.770_588_2,
            0.754_901_95,
            0.739_215_7,
            0.723_529_4,
            0.707_843_1,
            0.692_156_85,
            0.676_470_6,
            0.660_784_3,
            0.645_098_03,
            0.629_411_76,
            0.613_725_5,
            0.598_039_2,
            0.582_352_94,
            0.566_666_66,
            0.550_980_4,
            0.535_294_1,
            0.519_607_84,
            0.503_921_57,
            0.488_235_3,
            0.472_549_02,
            0.456_862_75,
            0.441_176_47,
            0.425_490_2,
            0.409_803_93,
            0.394_117_65,
            0.378_431_38,
            0.362_745_1,
            0.347_058_83,
            0.331_372_56,
            0.315_686_3,
            0.3,
            0.284_313_74,
            0.268_627_46,
            0.252_941_2,
            0.237_254_9,
            0.221_568_63,
            0.205_882_36,
            0.190_196_08,
            0.174_509_81,
            0.158_823_53,
            0.143_137_26,
            0.127_450_99,
            0.111_764_71,
            0.096_078_43,
            0.080_392_16,
            0.064_705_886,
            0.049_019_61,
            0.033_333_335,
            0.017_647_06,
            0.001_960_784_4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        pub const B: [f32; 256] = [
            0.5,
            0.515_686_3,
            0.531_372_55,
            0.547_058_8,
            0.562_745_1,
            0.578_431_37,
            0.594_117_64,
            0.609_803_9,
            0.625_490_2,
            0.641_176_46,
            0.656_862_74,
            0.672_549,
            0.688_235_3,
            0.703_921_56,
            0.719_607_83,
            0.735_294_1,
            0.750_980_4,
            0.766_666_65,
            0.782_352_9,
            0.798_039_2,
            0.813_725_5,
            0.829_411_75,
            0.845_098,
            0.860_784_3,
            0.876_470_57,
            0.892_156_84,
            0.907_843_1,
            0.923_529_4,
            0.939_215_66,
            0.954_901_93,
            0.970_588_2,
            0.986_274_5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.994_117_6,
            0.978_431_34,
            0.962_745_1,
            0.947_058_8,
            0.931_372_5,
            0.915_686_25,
            0.9,
            0.884_313_7,
            0.868_627_4,
            0.852_941_16,
            0.837_254_9,
            0.821_568_6,
            0.805_882_33,
            0.790_196_06,
            0.774_509_8,
            0.758_823_5,
            0.743_137_24,
            0.727_450_97,
            0.711_764_7,
            0.696_078_4,
            0.680_392_15,
            0.664_705_9,
            0.649_019_6,
            0.633_333_3,
            0.617_647_05,
            0.601_960_8,
            0.586_274_5,
            0.570_588_23,
            0.554_901_96,
            0.539_215_7,
            0.523_529_4,
            0.507_843_14,
            0.492_156_86,
            0.476_470_6,
            0.460_784_32,
            0.445_098_04,
            0.429_411_77,
            0.413_725_5,
            0.398_039_22,
            0.382_352_95,
            0.366_666_67,
            0.350_980_4,
            0.335_294_13,
            0.319_607_85,
            0.303_921_58,
            0.288_235_3,
            0.272_549_03,
            0.256_862_76,
            0.241_176_47,
            0.225_490_2,
            0.209_803_92,
            0.194_117_65,
            0.178_431_38,
            0.162_745_1,
            0.147_058_83,
            0.131_372_56,
            0.115_686_275,
            0.1,
            0.084_313_73,
            0.068_627_454,
            0.052_941_177,
            0.037_254_903,
            0.021_568_628,
            0.005_882_353,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
    }
}
