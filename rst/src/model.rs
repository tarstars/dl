//! Character-level vanilla RNN:
//!
//! ```text
//! h = tanh(Wxh·x + Whh·h + bh)
//! y = Why·h + by
//! ```
//!
//! `x` is a one-hot letter `(na, 1)`, `h` the hidden state `(nh, 1)`,
//! `y` the scores for the next letter `(na, 1)`.

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

/// A random index drawn with probabilities `p`.
fn draw(p: &Array2<f32>, rng: &mut StdRng) -> usize {
    let mut u: f32 = rng.random();
    for (i, &pi) in p.iter().enumerate() {
        u -= pi;
        if u < 0.0 {
            return i;
        }
    }
    p.len() - 1
}

/// What one forward step leaves behind for backprop.
struct Step {
    h_prev: Array2<f32>,
    h: Array2<f32>,
    y: Array2<f32>,
    p: Array2<f32>,
    tc: usize,
    tn: usize,
}


impl Rnn {
    /// Random weights, zero biases.
    pub fn new(na: usize, nh: usize, rng: &mut StdRng) -> Self {
        Rnn {
            na,
            nh,
            wxh: uniform(nh, na, na, rng),
            whh: uniform(nh, nh, nh, rng),
            why: uniform(na, nh, nh, rng),
            bh: Array2::zeros((nh, 1)),
            by: Array2::zeros((na, 1)),
        }
    }

    /// Total number of trainable numbers.
    pub fn n_params(&self) -> usize {
        [&self.wxh, &self.whh, &self.why, &self.bh, &self.by]
            .iter()
            .map(|m| m.len())
            .sum()
    }



    /// Plain SGD: `w -= lr * g` for each parameter, with every gradient
    /// element clipped to ±`clip` against exploding gradients.
    /// `grads` go in the order `wxh, whh, why, bh, by`.
    pub fn sgd_update(&mut self, lr: f32, clip: f32, grads: [&Array2<f32>; 5]) {
        let params = [&mut self.wxh, &mut self.whh, &mut self.why, &mut self.bh, &mut self.by];
        for (w, g) in params.into_iter().zip(grads) {
            w.scaled_add(-lr, &g.mapv(|v| v.clamp(-clip, clip)));
        }
    }

    /// Generates a name: starts from `start`, feeds each drawn letter back
    /// in, stops at `end` or after `max_len` letters. `start` and `end`
    /// are not included in the result.
    pub fn sample(&self, start: usize, end: usize, max_len: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut h = Array2::zeros((self.nh, 1));
        let mut tc = start;
        let mut name = Vec::new();
        for _ in 0..max_len {
            let h_raw = &self.whh.dot(&h) + &self.bh + self.wxh.slice(s![.., tc..tc+1]);
            h = h_raw.mapv(f32::tanh);
            let p = softmax(&(&self.why.dot(&h) + &self.by));
            tc = draw(&p, rng);
            if tc == end {
                break;
            }
            name.push(tc);
        }
        name
    }

    /// Forward step.
    pub fn forward_step(&self, h: &Array2<f32>, tc: usize, tn: usize) -> (Array2<f32>, f32) {
        let h_next_raw = &self.whh.dot(h) + &self.bh + self.wxh.slice(s![.., tc..tc+1]);
        let h_next = h_next_raw.mapv(f32::tanh);
        let y = &self.why.dot(&h_next) + &self.by;
        let p = softmax(&y);
        let loss = -p[[tn, 0]].ln();
        (h_next, loss)
    }

    pub fn forward_sequence(&self, it: &[usize]) -> (Array2<f32>, f32) {
        let mut h = Array2::zeros((self.nh, 1));
        let mut loss = 0.;

        for w in it.windows(2) {
            let (tc, tn) = (w[0], w[1]);
            let (h_next, delta) = self.forward_step(&h, tc, tn);
            h = h_next;
            loss += delta;
        }

        (h, loss)
    }

    pub fn train_step(&self, it: &[usize]) -> (f32, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)  {
        let mut h = Array2::zeros((self.nh, 1));
        let mut loss = 0.;
        let mut history = Vec::<Step>::new();


        for w in it.windows(2) {
            let (tc, tn) = (w[0], w[1]);
            let h_next_raw = &self.whh.dot(&h) + &self.bh + self.wxh.slice(s![.., tc..tc+1]);
            let h_next = h_next_raw.mapv(f32::tanh);
            let y = &self.why.dot(&h_next) + &self.by;
            let p = softmax(&y);            
            loss += -p[[tn, 0]].ln();
            history.push(Step { h_prev: h, h: h_next.clone(), y, p, tc, tn });
            h = h_next;
        }

        let mut dwxh = Array2::<f32>::zeros(self.wxh.raw_dim());
        let mut dwhh = Array2::<f32>::zeros(self.whh.raw_dim());
        let mut dwhy = Array2::<f32>::zeros(self.why.raw_dim());
        let mut dbh = Array2::<f32>::zeros(self.bh.raw_dim());
        let mut dby = Array2::<f32>::zeros(self.by.raw_dim());
        let mut dhnext = Array2::<f32>::zeros((self.nh, 1));

        for state in history.iter().rev() {
            let mut dy = state.p.clone();
            dy[[state.tn, 0]] -= 1.0;

            dwhy = dwhy + dy.dot(&state.h.t());
            dby = dby + &dy;
            let dh = self.why.t().dot(&dy) + &dhnext;
            let dpretanh = dh * (1.0 - &state.h.mapv(|x| x*x));
            dbh = dbh + &dpretanh;
            let mut dwxh_col = dwxh.slice_mut(s![.., state.tc..state.tc+1]);
            dwxh_col += &dpretanh;
            dwhh = dwhh + dpretanh.dot(&state.h_prev.t());
            dhnext = self.whh.t().dot(&dpretanh);

        }

        (loss, dwxh, dwhh, dwhy, dbh, dby, h)
    }
}
