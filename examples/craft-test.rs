use burn::backend::{CudaJit, Wgpu};
use burn::{
    module::Module,
    prelude::Backend,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::{Parser, ValueEnum};
use craft_burn::{
    craft::{
        utils::{adjust_coordinates, get_det_boxes},
        Craft, CraftRecord,
    },
    image_util::{
        float_to_color_map, resize_aspect_ratio, NormalizeMeanVarianceConfig, ResizeResult,
    },
};
use image::{DynamicImage, Rgb};
use imageproc::drawing::draw_hollow_polygon_mut;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use strum::Display;

#[derive(Parser, Debug)]
pub struct Args {
    /// Model weight file
    #[arg(long, default_value = "weights/craft_weights_half.mpk")]
    trained_model: PathBuf,
    /// The burn backend to use.
    #[arg(short, long, default_value_t = BurnBackend::Wgpu)]
    backend: BurnBackend,
    /// Confidence threshold for text detection
    #[arg(long, default_value_t = 0.7)]
    text_threshold: f64,
    /// Confidence threshold for links
    #[arg(long, default_value_t = 0.4)]
    link_threshold: f64,
    /// Cutoff threshold for text box
    #[arg(long, default_value_t = 0.4)]
    low_text: f64,
    /// Maximum side length for scaled image
    #[arg(long, default_value_t = 1280)]
    max_size: usize,
    /// Magnification ratio for input image
    #[arg(long, default_value_t = 1.5)]
    mag_ratio: f32,
    /// Test image
    #[arg(short = 'i', long, default_value = "test_images/test_1.png")]
    test_image: PathBuf,
    /// Test image
    #[arg(short = 'o', long, default_value = "result/")]
    out_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, Display, ValueEnum)]
pub enum BurnBackend {
    #[strum(serialize = "wgpu")]
    Wgpu,
    #[strum(serialize = "cuda")]
    Cuda,
    #[strum(serialize = "tch")]
    Tch,
    #[strum(serialize = "tch-gpu")]
    TchGpu,
}

fn main() {
    let args = Args::parse();

    // Import from official weights

    // let device = Default::default();
    // let load_args = LoadArgs::new("./trained/weights.pt".into())
    //     .with_key_remap(r"upconv([0-9])\.conv\.0", "upconv$1.conv1")
    //     .with_key_remap(r"upconv([0-9])\.conv\.1", "upconv$1.batch_norm1")
    //     .with_key_remap(r"upconv([0-9])\.conv\.3", "upconv$1.conv2")
    //     .with_key_remap(r"upconv([0-9])\.conv\.4", "upconv$1.batch_norm2")
    //     .with_key_remap(r"basenet\.slice([0-9])\.([0-9])", "basenet.slice_$1.feat$2")
    //     .with_debug_print();
    // let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    // let record: CraftRecord<NdArray<f32>> = recorder
    //     .load(load_args, &device)
    //     .expect("Should decode state successfully");

    // let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    // recorder
    //     .record(record, "craft_weights".into())
    //     .expect("Failed to save model record");

    match args.backend {
        BurnBackend::Wgpu => {
            type MyBackend = Wgpu<half::f16, i32>;

            let device = Default::default();

            println!("Using wgpu");
            run::<MyBackend>(&device, args);
        }
        BurnBackend::Cuda => {
            type MyBackend = CudaJit<half::f16, i32>;

            let device = Default::default();

            println!("Using CUDA");
            run::<MyBackend>(&device, args);
        }
        BurnBackend::Tch => {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};
            type MyBackend = LibTorch<half::f16>;

            let device = LibTorchDevice::Cpu;

            println!("Using tch");
            run::<MyBackend>(&device, args);
        }
        BurnBackend::TchGpu => {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};
            type MyBackend = LibTorch<half::f16>;

            let device = LibTorchDevice::Cuda(0);

            println!("Using tch-gpu");
            run::<MyBackend>(&device, args);
        }
    }
}

pub fn test_net<B: Backend>(
    net: Craft<B>,
    image: DynamicImage,
    text_threshold: f64,
    link_threshold: f64,
    low_text: f64,
    out_dir: &Path,
    device: &B::Device,
) {
    let mut image_out = image.to_rgb8();
    let total_start = Instant::now();
    let ResizeResult::<B> { image, ratio, .. } = resize_aspect_ratio(image, 1280, 1.5, device);

    let ratio_h = 1.0 / ratio;
    let ratio_w = ratio_h;

    let start = Instant::now();

    let norm_mean_variance = NormalizeMeanVarianceConfig::new().init(device);

    let x = norm_mean_variance.forward(image);

    let (y, _feature) = net.forward(x);

    let score_text = y.clone().narrow(3, 0, 1);
    let score_link = y.clone().narrow(3, 1, 1);

    // Sync to properly measure time of each step
    let _ = score_text.to_data();
    let _ = score_link.to_data();

    let net_time = Instant::now() - start;
    println!("Network took {:?}", net_time);

    let image_text = float_to_color_map(score_text.clone());
    let image_link = float_to_color_map(score_link.clone());

    image_text.save(out_dir.join("image_text.png")).unwrap();
    image_link.save(out_dir.join("image_link.png")).unwrap();

    let start = Instant::now();
    let boxes = get_det_boxes(
        score_text.clone(),
        score_link.clone(),
        text_threshold,
        link_threshold,
        low_text,
    );

    let boxes = adjust_coordinates(boxes, ratio_w, ratio_h);
    println!("Processing time: {:?}", Instant::now() - start);
    println!("Total time: {:?}", Instant::now() - total_start);

    for bbox in boxes.values() {
        draw_hollow_polygon_mut(&mut image_out, bbox, Rgb([255, 0, 0]));
    }
    DynamicImage::ImageRgb8(image_out)
        .save(out_dir.join("boxes.png"))
        .unwrap();
}

pub fn run<B: Backend>(device: &B::Device, args: Args) {
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    let record: CraftRecord<B> = recorder
        .load(args.trained_model, &Default::default())
        .expect("Failed to load model");

    let net = Craft::<B>::init(device).load_record(record);
    let image = image::open(args.test_image).unwrap();

    fs::create_dir_all(&args.out_dir).unwrap();

    // Run twice so we can get warmed up execution times as well as cold starts
    test_net(
        net.clone(),
        image.clone(),
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        &args.out_dir,
        device,
    );
    test_net(
        net.clone(),
        image.clone(),
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        &args.out_dir,
        device,
    );
}
