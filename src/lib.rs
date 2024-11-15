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
