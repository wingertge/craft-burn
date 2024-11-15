use burn::backend::{CudaJit, Wgpu};
use craft_burn::test_craft::run;

fn main() {
    use std::env;

    let backend = env::args().nth(1).unwrap_or_else(|| "cuda".to_string());

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

    if backend == "wgpu" {
        type MyBackend = Wgpu<half::f16, i32>;

        let device = Default::default();

        println!("Using wgpu");
        run::<MyBackend>(&device);
    } else if backend == "tch" {
        use burn::backend::{libtorch::LibTorchDevice, LibTorch};
        type MyBackend = LibTorch<half::f16>;

        let device = LibTorchDevice::Cuda(0);

        println!("Using tch");
        run::<MyBackend>(&device);
    } else {
        type MyBackend = CudaJit<half::f16, i32>;

        let device = Default::default();

        println!("Using CUDA");
        run::<MyBackend>(&device);
    }
}
