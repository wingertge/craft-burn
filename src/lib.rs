pub mod craft;
pub mod image_util;
pub mod test_craft;

#[macro_export]
macro_rules! sequential {
    ($x:expr, $($module:expr),*) => {
        {
            let x = $x;
            $(
                let x = $module.forward(x);
            )*
            x
        }
    };
}

#[macro_export]
macro_rules! stats {
    ($name:literal, $x:expr) => {{
        print!($name);
        println!(
            " min: {}, max: {}, mean: {}",
            $x.clone().min().into_scalar(),
            $x.clone().max().into_scalar(),
            $x.clone().mean().into_scalar()
        )
    }};
}
