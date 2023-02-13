//! From https://github.com/mthh/rbf_interp

#![crate_name = "rbf_interp"]
#![crate_type = "lib"]
#![deny(
    trivial_casts,
    trivial_numeric_casts,
    missing_debug_implementations,
    missing_copy_implementations,
    unstable_features,
    unsafe_code,
    unused_import_braces
)]
// missing_docs, unused_qualifications

#[macro_use]
extern crate error_chain;

extern crate num_traits;
extern crate rulinalg;

mod bbox;
pub mod errors;
mod rbf;
mod utils;

pub use self::bbox::Bbox;
pub use self::rbf::{rbf_interpolation, Rbf};
pub use self::utils::PtValue;

#[cfg(test)]
mod test;
