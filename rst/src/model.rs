//! Character-level vanilla RNN:
//!
//!     h = tanh(Wxh·x + Whh·h + bh)
//!     y = Why·h + by
//!
//! `x` is a one-hot letter `(n_in, 1)`, `h` the hidden state `(n_hidden, 1)`,
//! `y` the scores for the next letter `(n_in, 1)`.

use ndarray::{Array2, s};
use rand::RngExt;
use rand::rngs::StdRng;

pub struct Rnn {
    pub na: usize,
    pub nh: usize,
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

fn softmax(y: &Array2<f32>) -> Array2<f32> {
    let max = y.fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let e = y.mapv(|v| (v - max).exp());
    let sum = e.sum();
    e / sum
}


impl Rnn {
    /// Random weights, zero biases.
    pub fn new(n_in: usize, n_hidden: usize, rng: &mut StdRng) -> Self {
        Rnn {
            na: n_in,
            nh: n_hidden,
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



    /// Forward step.
    pub fn forward_step(&self, h: &Array2<f32>, t: usize) -> (Array2<f32>, f32) {
        let h_next_raw = &self.whh.dot(h) + &self.bh + self.wxh.slice(s![.., t..t+1]);
        let h_next = h_next_raw.mapv(f32::tanh);
        let y = &self.why.dot(&h_next) + &self.by;
        let p = softmax(&y);
        let loss = -p[[t, 0]].ln();
        (h_next, loss)
    }

    pub fn forward_sequence(&self, it: impl Iterator<Item = usize>) -> (Array2<f32>, f32) {
        let mut h = Array2::zeros((self.nh, 1));
        let mut loss = 0.;
        let mut delta = 0.;

        for c in it {
            (h, delta) = self.forward_step( &h, c);
            loss += delta;
        }

        (h, loss)
    }

    pub fn train_step(&self, lr: f32, it: impl Iterator<Item = usize>) {
        let mut h = Array2::zeros((self.nh, 1));
        let mut loss = 0.;
        let mut delta = 0.;

        for c in it {
            let h_next_raw = &self.whh.dot(&h) + &self.bh + self.wxh.slice(s![.., c..c+1]);
            let h_next = h_next_raw.mapv(f32::tanh);
            let y = &self.why.dot(&h_next) + &self.by;
            let p = softmax(&y);            
            loss += -p[[c, 0]].ln();
        }

    }
}
