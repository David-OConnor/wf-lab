//! This module contains a `Complex` number type, and methods for it.

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

pub const IM: Cplx = Cplx::new(0., 1.);

#[derive(Copy, Clone)]
pub struct Cplx {
    // todo should probably just use num::Complex
    pub real: f64,
    pub im: f64,
}

impl Cplx {
    pub fn new(real: f64, im: f64) -> Self {
        Self { real, im }
    }

    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            im: -self.im,
        }
    }

    pub fn mag(&self) -> f64 {
        (self.real.powi(2) + self.im.powi(2)).sqrt()
    }

    pub fn phase(&self) -> f64 {
        (self.im).atan2(self.real)
    }
}

impl Add for Cplx {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            im: self.im + other.im,
        }
    }
}

impl Sub for Cplx {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            im: self.im - other.im,
        }
    }
}

impl Mul for Cplx {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.im * other.im,
            im: self.real * other.im + self.im * other.real,
        }
    }
}

impl fmt::Display for Cplx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}i", self.real, self.im)
    }
}

// todo impl Div for Cplx.
