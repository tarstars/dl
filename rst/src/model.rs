//! Character-level vanilla RNN:
//!
//!     h = tanh(Wxh·x + Whh·h + bh)
//!     y = Why·h + by
//!
//! `x` is a one-hot letter `(n_in, 1)`, `h` the hidden state `(n_hidden, 1)`,
//! `y` the scores for the next letter `(n_in, 1)`.

use ndarray::Array2;
use rand::RngExt;
use rand::rngs::StdRng;

pub struct Rnn {
    pub wxh: Array2<f32>,
    pub whh: Array2<f32>,
    pub why: Array2<f32>,
    pub bh: Array2<f32>,
    pub by: Array2<f32>,
}

/// A `rows × cols` matrix, uniform in ±1/√fan_in.
fn uniform(rows: usize, cols: usize, fan_in: usize, rng: &mut StdRng) -> Array2<f32> {
    let s = 1.0 / (fan_in as f32).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.random_range(-s..s))
}

impl Rnn {
    /// Random weights, zero biases.
    pub fn new(n_in: usize, n_hidden: usize, rng: &mut StdRng) -> Self {
        Rnn {
            wxh: uniform(n_hidden, n_in, n_in, rng),
            whh: uniform(n_hidden, n_hidden, n_hidden, rng),
            why: uniform(n_in, n_hidden, n_hidden, rng),
            bh: Array2::zeros((n_hidden, 1)),
            by: Array2::zeros((n_in, 1)),
        }
    }

    /// Total number of trainable numbers.
    pub fn n_params(&self) -> usize {
        [&self.wxh, &self.whh, &self.why, &self.bh, &self.by]
            .iter()
            .map(|m| m.len())
            .sum()
    }
}
