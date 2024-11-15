use std::collections::HashMap;

use burn::{prelude::Backend, tensor::Tensor};
use connected::{connected_components_with_stats, ConnectedComponentsResult};
use float_ord::FloatOrd;
use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use imageproc::{
    contours::find_contours, definitions::Image, distance_transform::Norm, geometry::min_area_rect,
    morphology::dilate_mut, point::Point, region_labelling::Connectivity,
};

pub fn adjust_coordinates(
    mut boxes: HashMap<u32, [Point<f32>; 4]>,
    ratio_w: f32,
    ratio_h: f32,
) -> HashMap<u32, [Point<f32>; 4]> {
    for bbox in boxes.values_mut().flatten() {
        bbox.x *= ratio_w * 2.0;
        bbox.y *= ratio_h * 2.0;
    }
    boxes
}

type FloatGrayImage = ImageBuffer<Luma<f32>, Vec<f32>>;

pub fn get_det_boxes<B: Backend>(
    text_map: Tensor<B, 4>,
    link_map: Tensor<B, 4>,
    text_threshold: f64,
    link_threshold: f64,
    low_text: f64,
) -> HashMap<u32, [Point<f32>; 4]> {
    let shape = text_map.shape().dims::<4>();
    let [_, height, width, _] = shape;

    let text_score = text_map.clone().greater_equal_elem(low_text).int();
    let link_score = link_map.clone().greater_equal_elem(link_threshold).int();

    let combined = (text_score.clone() + link_score.clone()).clamp(0, 1);
    let data = combined.to_data().convert::<u8>().to_vec::<u8>().unwrap();
    let text_score_comb = GrayImage::from_vec(width as u32, height as u32, data).unwrap();

    let ConnectedComponentsResult {
        stats,
        num_labels,
        labels,
    } = connected_components_with_stats(&text_score_comb, Connectivity::Four, Luma([0]));
    let text_map_data = text_map.into_data().convert::<f32>().to_vec::<f32>();
    let text_map =
        FloatGrayImage::from_vec(width as u32, height as u32, text_map_data.unwrap()).unwrap();

    let text_score = text_score.into_data().convert::<u8>().to_vec::<u8>();
    let text_score = GrayImage::from_vec(width as u32, height as u32, text_score.unwrap()).unwrap();
    let link_score = link_score.into_data().convert::<u8>().to_vec::<u8>();
    let link_score = GrayImage::from_vec(width as u32, height as u32, link_score.unwrap()).unwrap();

    let mut boxes = HashMap::new();

    unsafe {
        for k in 1..num_labels {
            let size = stats.area[k as usize];
            if size < 10 {
                continue;
            }

            let max = text_map.enumerate_pixels().filter_map(|(x, y, p)| {
                (labels.unsafe_get_pixel(x, y) == Luma([k])).then_some(FloatOrd(p.0[0]))
            });
            let max = max.max().unwrap().0;

            if max < text_threshold as f32 {
                continue;
            }

            let mut seg_map = seg_map(&labels, k, &text_score, &link_score);
            let x = stats.left[k as usize];
            let y = stats.top[k as usize];
            let w = stats.right[k as usize] - x;
            let h = stats.bottom[k as usize] - y;

            let niter = ((size as f32 * w.min(h) as f32 / (w * h) as f32).sqrt() * 2.0) as u32;
            dilate_mut(&mut seg_map, Norm::L1, (1 + niter) as u8);

            let contours = find_contours::<i32>(&seg_map);
            let [a, b, c, d] = min_area_rect(&contours[0].points);

            let as_float = |p: Point<i32>| Point {
                x: p.x as f32,
                y: p.y as f32,
            };

            boxes.insert(k, [as_float(a), as_float(b), as_float(c), as_float(d)]);
        }
    }

    boxes
}

fn seg_map(
    labels: &Image<Luma<u32>>,
    group: u32,
    text_score: &GrayImage,
    link_score: &GrayImage,
) -> GrayImage {
    let mut out = GrayImage::new(labels.width(), labels.height());
    let inputs = labels.iter().zip(text_score.iter()).zip(link_score.iter());
    for (((g, text), link), p) in inputs.zip(out.iter_mut()) {
        let link_only = *text == 0 && *link == 1;
        *p = if *g == group && !link_only { 255 } else { 0 };
    }
    out
}

mod connected {
    use std::cmp;

    use image::{GenericImage, GenericImageView, ImageBuffer, Luma};
    use imageproc::{
        definitions::Image, region_labelling::Connectivity, union_find::DisjointSetForest,
    };

    #[derive(Default)]
    pub struct Stats {
        pub left: Vec<u32>,
        pub top: Vec<u32>,
        pub right: Vec<u32>,
        pub bottom: Vec<u32>,
        pub area: Vec<u32>,
    }

    pub struct ConnectedComponentsResult {
        pub stats: Stats,
        pub num_labels: u32,
        pub labels: Image<Luma<u32>>,
    }

    pub fn connected_components_with_stats<I>(
        image: &I,
        conn: Connectivity,
        background: I::Pixel,
    ) -> ConnectedComponentsResult
    where
        I: GenericImage,
        I::Pixel: Eq,
    {
        let (width, height) = image.dimensions();
        let image_size = width as usize * height as usize;
        if image_size >= 2usize.saturating_pow(32) {
            panic!("Images with 2^32 or more pixels are not supported");
        }

        let mut out = ImageBuffer::new(width, height);

        // TODO: add macro to abandon early if either dimension is zero
        if width == 0 || height == 0 {
            return ConnectedComponentsResult {
                stats: Stats::default(),
                num_labels: 0,
                labels: out,
            };
        }

        let mut forest = DisjointSetForest::new(image_size);
        let mut adj_labels = [0u32; 4];
        let mut next_label = 1;

        for y in 0..height {
            for x in 0..width {
                let current = unsafe { image.unsafe_get_pixel(x, y) };
                if current == background {
                    continue;
                }

                let mut num_adj = 0;

                if x > 0 {
                    // West
                    let pixel = unsafe { image.unsafe_get_pixel(x - 1, y) };
                    if pixel == current {
                        let label = unsafe { out.unsafe_get_pixel(x - 1, y)[0] };
                        adj_labels[num_adj] = label;
                        num_adj += 1;
                    }
                }

                if y > 0 {
                    // North
                    let pixel = unsafe { image.unsafe_get_pixel(x, y - 1) };
                    if pixel == current {
                        let label = unsafe { out.unsafe_get_pixel(x, y - 1)[0] };
                        adj_labels[num_adj] = label;
                        num_adj += 1;
                    }

                    if conn == Connectivity::Eight {
                        if x > 0 {
                            // North West
                            let pixel = unsafe { image.unsafe_get_pixel(x - 1, y - 1) };
                            if pixel == current {
                                let label = unsafe { out.unsafe_get_pixel(x - 1, y - 1)[0] };
                                adj_labels[num_adj] = label;
                                num_adj += 1;
                            }
                        }
                        if x < width - 1 {
                            // North East
                            let pixel = unsafe { image.unsafe_get_pixel(x + 1, y - 1) };
                            if pixel == current {
                                let label = unsafe { out.unsafe_get_pixel(x + 1, y - 1)[0] };
                                adj_labels[num_adj] = label;
                                num_adj += 1;
                            }
                        }
                    }
                }

                if num_adj == 0 {
                    unsafe {
                        out.unsafe_put_pixel(x, y, Luma([next_label]));
                    }
                    next_label += 1;
                } else {
                    let mut min_label = u32::MAX;
                    for n in 0..num_adj {
                        min_label = cmp::min(min_label, adj_labels[n]);
                    }
                    unsafe {
                        out.unsafe_put_pixel(x, y, Luma([min_label]));
                    }
                    for n in 0..num_adj {
                        forest.union(min_label as usize, adj_labels[n] as usize);
                    }
                }
            }
        }

        // Make components start at 1
        let mut output_labels = vec![0u32; image_size];
        let mut count = 1;
        let forest_count = forest.num_trees();
        let mut stats = Stats {
            left: vec![u32::MAX; forest_count],
            top: vec![u32::MAX; forest_count],
            right: vec![0u32; forest_count],
            bottom: vec![0u32; forest_count],
            area: vec![0u32; forest_count],
        };

        unsafe {
            for y in 0..height {
                for x in 0..width {
                    let label = {
                        if image.unsafe_get_pixel(x, y) == background {
                            continue;
                        }
                        out.unsafe_get_pixel(x, y)[0]
                    };
                    let root = forest.root(label as usize);
                    let mut output_label = *output_labels.get_unchecked(root);
                    if output_label < 1 {
                        output_label = count;
                        count += 1;
                    }
                    *output_labels.get_unchecked_mut(root) = output_label;
                    out.unsafe_put_pixel(x, y, Luma([output_label]));
                    let label = output_label as usize;
                    let left = stats.left.get_unchecked_mut(label);
                    let right = stats.right.get_unchecked_mut(label);
                    let top = stats.top.get_unchecked_mut(label);
                    let bottom = stats.bottom.get_unchecked_mut(label);
                    *left = (*left).min(x);
                    *top = (*top).min(y);
                    *right = (*right).max(x);
                    *bottom = (*bottom).max(y);
                    *stats.area.get_unchecked_mut(label) += 1;
                }
            }
        }

        ConnectedComponentsResult {
            stats,
            num_labels: count,
            labels: out,
        }
    }
}
