use std::path::Path;

use burn::{
    prelude::Backend,
    record::{FullPrecisionSettings, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use crate::{refine::RefineNetRecord, CraftRecord};

/// Load pytorch weights. Requires reexport from the python version because the official weights use
/// an outdated file format
pub fn load_pytorch_weights<B: Backend>(
    weights: impl AsRef<Path>,
    device: &B::Device,
) -> CraftRecord<B> {
    let load_args = LoadArgs::new(weights.as_ref().into())
        .with_key_remap(r"module\.(.+)", "$1")
        .with_key_remap(r"upconv([0-9])\.conv\.0", "upconv$1.conv1")
        .with_key_remap(r"upconv([0-9])\.conv\.1", "upconv$1.batch_norm1")
        .with_key_remap(r"upconv([0-9])\.conv\.3", "upconv$1.conv2")
        .with_key_remap(r"upconv([0-9])\.conv\.4", "upconv$1.batch_norm2")
        .with_key_remap(r"basenet\.slice([0-9])\.([0-9])", "basenet.slice_$1.feat$2");
    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .load(load_args, device)
        .expect("Should decode state successfully")
}

/// Load pytorch weights for the refiner network. Requires reexport from the python version because
/// the official weights use an outdated file format
pub fn load_refiner_weights<B: Backend>(
    weights: impl AsRef<Path>,
    device: &B::Device,
) -> RefineNetRecord<B> {
    let load_args = LoadArgs::new(weights.as_ref().into())
        .with_key_remap(r"module\.(.+)\.([0-9])", "$1.feat$2");
    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .load(load_args, device)
        .expect("Should decode state successfully")
}
