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

#[derive(Clone)]
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

/// A random index drawn with probabilities proportional to `p`
/// (`p` need not sum to 1).
fn draw(p: &Array2<f32>, rng: &mut StdRng) -> usize {
    let mut u: f32 = rng.random::<f32>() * p.sum();
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

    /// One step: the next hidden state and the next-letter probabilities
    /// `softmax(y / temperature)`. Temperature below 1 sharpens the
    /// distribution, above 1 flattens it.
    fn probs(&self, h: &Array2<f32>, tc: usize, temperature: f32) -> (Array2<f32>, Array2<f32>) {
        let h_raw = &self.whh.dot(h) + &self.bh + self.wxh.slice(s![.., tc..tc + 1]);
        let h_next = h_raw.mapv(f32::tanh);
        let y = &self.why.dot(&h_next) + &self.by;
        (h_next, softmax(&(y / temperature)))
    }

    /// Feeds `start` and then every letter of `prefix`. Returns the hidden
    /// state and the next-letter probabilities after the last one.
    fn read_prefix(&self, start: usize, prefix: &[usize], temperature: f32) -> (Array2<f32>, Array2<f32>) {
        let (mut h, mut p) = self.probs(&Array2::zeros((self.nh, 1)), start, temperature);
        for &tc in prefix {
            (h, p) = self.probs(&h, tc, temperature);
        }
        (h, p)
    }

    /// The probability of each alphabet letter coming right after `prefix`.
    pub fn next_probs(&self, start: usize, prefix: &[usize], temperature: f32) -> Vec<f32> {
        self.read_prefix(start, prefix, temperature).1.iter().copied().collect()
    }

    /// Generates a name that begins with `prefix`: feeds `start` and the
    /// prefix, then draws letters and feeds each one back in. Stops at `end`
    /// or when the name has `max_len` letters. The result includes the
    /// prefix and never contains `start` or `end`.
    pub fn sample(
        &self,
        start: usize,
        end: usize,
        prefix: &[usize],
        max_len: usize,
        temperature: f32,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let (mut h, mut p) = self.read_prefix(start, prefix, temperature);
        let mut name = prefix.to_vec();
        while name.len() < max_len {
            p[[start, 0]] = 0.0;
            let tc = draw(&p, rng);
            if tc == end {
                break;
            }
            name.push(tc);
            (h, p) = self.probs(&h, tc, temperature);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const START: usize = 0;
    const END: usize = 1;

    /// A small model: 5 letters, hidden state of 4.
    fn tiny() -> Rnn {
        Rnn::new(5, 4, &mut StdRng::seed_from_u64(7))
    }

    #[test]
    fn next_probs_sum_to_one() {
        let rnn = tiny();
        for t in [0.5, 1.0, 2.0] {
            for prefix in [&[][..], &[2, 3][..]] {
                let sum: f32 = rnn.next_probs(START, prefix, t).iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "sum {sum} at T={t}");
            }
        }
    }

    #[test]
    fn low_temperature_concentrates_on_argmax() {
        let mut rnn = tiny();
        rnn.by[[3, 0]] = 10.0; // letter 3 has by far the highest score
        let cold = rnn.next_probs(START, &[2], 0.05);
        let warm = rnn.next_probs(START, &[2], 1.0);
        assert!(cold[3] > 0.99, "cold p[3] = {}", cold[3]);
        assert!(warm[3] < cold[3]);
    }

    #[test]
    fn sample_keeps_prefix_and_skips_markers() {
        let rnn = tiny();
        let mut rng = StdRng::seed_from_u64(1);
        let prefix = [2, 3];
        for _ in 0..200 {
            let name = rnn.sample(START, END, &prefix, 10, 1.0, &mut rng);
            assert!(name.starts_with(&prefix));
            assert!(name.len() <= 10);
            assert!(!name.contains(&START) && !name.contains(&END), "{name:?}");
        }
    }

    #[test]
    fn sample_with_prefix_at_max_len_returns_prefix() {
        let rnn = tiny();
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(rnn.sample(START, END, &[2, 3, 4], 3, 1.0, &mut rng), vec![2, 3, 4]);
    }
}
