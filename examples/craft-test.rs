use burn::{
    backend::{CudaJit, Wgpu},
    record::FullPrecisionSettings,
};
use burn::{
    module::Module,
    prelude::Backend,
    record::{NamedMpkFileRecorder, Recorder},
};
use clap::{Parser, ValueEnum};
use craft_burn::{
    image_util::{
        float_to_color_map, resize_aspect_ratio, NormalizeMeanVarianceConfig, ResizeResult,
    },
    loader,
    refine::{RefineNet, RefineNetRecord},
    utils::{adjust_coordinates, get_det_boxes},
    Craft, CraftRecord,
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
    #[arg(long, default_value = "weights/craft_mlt_25k.mpk")]
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
    #[arg(short = 'o', long, default_value = "result")]
    out_dir: PathBuf,
    /// Whether to use refiner net
    #[arg(long, default_value_t = false)]
    refine: bool,
    /// Path to refiner weights
    #[arg(long, default_value = "weights/craft_refiner_CTW1500.mpk")]
    refiner_model: PathBuf,

    /// Convert pytorch weights to mpk
    #[arg(long)]
    convert: bool,
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
            type MyBackend = LibTorch<f32>;

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

#[allow(clippy::too_many_arguments)]
pub fn test_net<B: Backend>(
    net: Craft<B>,
    refine_net: Option<RefineNet<B>>,
    img_name: &str,
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

    let (y, feature) = net.forward(x);

    let score_text = y.clone().narrow(3, 0, 1);
    let mut score_link = y.clone().narrow(3, 1, 1);

    // Sync to properly measure time of each step
    let _ = score_text.to_data();
    let _ = score_link.to_data();

    let net_time = Instant::now() - start;
    println!("Network took {:?}", net_time);

    if let Some(refine_net) = refine_net {
        let start = Instant::now();
        score_link = refine_net.forward(y.clone(), feature).narrow(3, 0, 1);
        let _ = score_link.to_data();
        println!("RefineNet took {:?}", Instant::now() - start);
    }

    let image_text = float_to_color_map(score_text.clone());
    let image_link = float_to_color_map(score_link.clone());

    image_text
        .save(out_dir.join(format!("{img_name}_text.png")))
        .unwrap();
    image_link
        .save(out_dir.join(format!("{img_name}_link.png")))
        .unwrap();

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
        .save(out_dir.join(format!("{img_name}_boxes.png")))
        .unwrap();
}

pub fn run<B: Backend>(device: &B::Device, mut args: Args) {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    if args.convert {
        let record = loader::load_pytorch_weights::<B>(&args.trained_model, device);
        args.trained_model.set_extension("mpk");
        recorder.record(record, args.trained_model.clone()).unwrap();

        if args.refine {
            let record = loader::load_refiner_weights::<B>(&args.refiner_model, device);
            args.refiner_model.set_extension("mpk");
            recorder.record(record, args.refiner_model.clone()).unwrap();
        }
    }
    let record: CraftRecord<B> = recorder
        .load(args.trained_model, &Default::default())
        .expect("Failed to load model");

    let net = Craft::<B>::init(device).load_record(record);
    let refine_net = args.refine.then(|| {
        let record: RefineNetRecord<B> = recorder
            .load(args.refiner_model, &Default::default())
            .expect("Failed to load model");
        RefineNet::init(device).load_record(record)
    });

    let mut image_name = args.test_image.clone();
    image_name.set_extension("");
    let image_name = image_name.file_name().unwrap().to_string_lossy();
    let image = image::open(args.test_image).unwrap();

    fs::create_dir_all(&args.out_dir).unwrap();

    // Run twice so we can get warmed up execution times as well as cold starts
    test_net(
        net.clone(),
        refine_net.clone(),
        &image_name,
        image.clone(),
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        &args.out_dir,
        device,
    );
    test_net(
        net.clone(),
        refine_net.clone(),
        &image_name,
        image.clone(),
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        &args.out_dir,
        device,
    );
}
