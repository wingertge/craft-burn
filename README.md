## CRAFT: Character-Region Awareness For Text detection

Burn implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/file/d/1aquT8wd3wp6CIJJ5bCXD4oEaU1yWopUZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**

Clova AI Research, NAVER Corp.

_Adapted by [Genna Wingert](https://github.com/wingertge)_

### Sample Results

### Overview

Burn implementation for CRAFT text detector that effectively detect text area by exploring each
character region and affinity between characters. The bounding box of texts are obtained by simply
finding minimum bounding rectangles on binary map after thresholding character region and affinity
scores.

Adapted from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch/)

The refiner currently isn't implemented, and `cubecl` backends won't work with burn main until
[this fix](https://github.com/tracel-ai/burn/pull/2499)
have been merged.

## Getting started

### Training

The code for training is not included in this repository, as the original authors cannot release the full training code for IP reason.

### Test instruction using pretrained model

-   Download the trained models (converted because originals use a legacy format)
    | _Model name_ | _Used datasets_ | _Languages_ | _Purpose_ | _Model Link_ |
    | :----------- | :-------------------- | :---------- | :-------------------------- | :-------------------------------------------------------------------------- |
    | General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/file/d/1aquT8wd3wp6CIJJ5bCXD4oEaU1yWopUZ) |
    | IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/file/d/1xo_XrpNibEnkREJQofUvd3D_onggsfeP) |
    | LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/file/d/19gd0ujol534mqeCpce2pWafTrzXATCDD) |

*   Run with pretrained model

```
cargo run --example test-craft --release --trained_model=[weightfile] --test_image=[path to test image]
```

The result image and socre maps will be saved to `./result` by default.

### Arguments

-   `--trained_model`: pretrained model
-   `--text_threshold`: text confidence threshold
-   `--low_text`: text low-bound score
-   `--link_threshold`: link confidence threshold
-   `--backend`: backend to use for inference (default: `wgpu`)
-   `--max_size`: max image size for inference
-   `--mag_ratio`: image magnification ratio
-   `--test_file`: file path to input image
-   `--refine`: use link refiner for sentense-level dataset
-   `--refiner_model`: pretrained refiner model

## Links

-   Original implementation: https://github.com/clovaai/CRAFT-pytorch
