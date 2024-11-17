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

mod craft;

pub mod image_util;
pub mod refine;
pub use craft::*;

#[cfg(feature = "import")]
pub mod loader;
