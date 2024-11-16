use std::time::Instant;

use burn::{
    module::Module,
    prelude::Backend,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use image::{DynamicImage, Rgb};
use imageproc::drawing::draw_hollow_polygon_mut;

use crate::{
    craft::{
        utils::{adjust_coordinates, get_det_boxes},
        Craft, CraftRecord,
    },
    image_util::{
        float_to_color_map, resize_aspect_ratio, NormalizeMeanVarianceConfig, ResizeResult,
    },
};

pub fn test_net<B: Backend>(
    net: Craft<B>,
    image: DynamicImage,
    text_threshold: f64,
    link_threshold: f64,
    low_text: f64,
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

    image_text.save("result/image_text.png").unwrap();
    image_link.save("result/image_link.png").unwrap();

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
        .save("result/boxes.png")
        .unwrap();
}

pub fn run<B: Backend>(device: &B::Device) {
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    let record: CraftRecord<B> = recorder
        .load("weights/craft_weights_half.mpk".into(), &Default::default())
        .expect("Failed to load model");

    let net = Craft::<B>::init(device).load_record(record);
    let image = image::open("test_images/test_1.png").unwrap();

    test_net(net.clone(), image.clone(), 0.7, 0.4, 0.4, device);
    test_net(net.clone(), image.clone(), 0.7, 0.4, 0.4, device);
}
