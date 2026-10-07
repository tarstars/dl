//! Training runs for the web lab: parameters, progress points, the
//! background training loop, and probing a run's model. Knows nothing
//! about HTTP; `src/bin/server.rs` wraps it.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::data::{self, END, START};
use crate::model::Rnn;

/// Validation loss at each point is computed on this many names.
pub const VAL_SUBSET: usize = 1000;
/// Generated names are cut off at this many letters; longer prefixes are refused.
pub const MAX_NAME_LEN: usize = 40;
/// Generated names stored with each point.
const N_SAMPLES: usize = 3;
/// How many letters the probability panel shows.
pub const TOP_PROBS: usize = 10;

/// The data every run trains on, encoded as alphabet indices.
pub struct Dataset {
    pub alphabet: Vec<char>,
    pub char2ind: HashMap<char, usize>,
    pub train: Vec<Vec<usize>>,
    pub val: Vec<Vec<usize>>,
}

impl Dataset {
    /// Loads the names, drops rare characters, builds the alphabet and splits
    /// train/val exactly like the CLI trainer does.
    pub fn load(dir: &str, min_char_count: usize, val_fraction: f64, seed: u64) -> crate::Result<Self> {
        let names = data::load_names(dir)?;
        let rare = data::rare_chars(&names, min_char_count);
        let names = data::drop_names_with(names, &rare);
        let (alphabet, char2ind) = data::build_alphabet(&names);
        let (train, val) = data::split_train_val(names, val_fraction, &mut StdRng::seed_from_u64(seed));
        let encode = |names: Vec<String>| names.iter().map(|n| data::encode(n, &char2ind)).collect();
        Ok(Dataset { train: encode(train), val: encode(val), alphabet, char2ind })
    }

    pub fn start(&self) -> usize {
        self.char2ind[&START]
    }

    pub fn end(&self) -> usize {
        self.char2ind[&END]
    }

    /// The prefix as alphabet indices, or an error naming the first character
    /// that is not a letter of the alphabet (the `START`/`END` markers are
    /// not letters).
    pub fn encode_prefix(&self, prefix: &str) -> Result<Vec<usize>, ProbeError> {
        if prefix.chars().count() > MAX_NAME_LEN {
            return Err(ProbeError::PrefixTooLong);
        }
        prefix
            .chars()
            .map(|c| match self.char2ind.get(&c) {
                Some(&i) if c != START && c != END => Ok(i),
                _ => Err(ProbeError::UnknownChar(c)),
            })
            .collect()
    }

    pub fn decode(&self, name: &[usize]) -> String {
        name.iter().map(|&i| self.alphabet[i]).collect()
    }

    /// Mean loss per predicted character over `seqs`, in nats.
    fn mean_loss(&self, rnn: &Rnn, seqs: &[Vec<usize>]) -> f32 {
        let loss: f32 = seqs.iter().map(|seq| rnn.forward_sequence(seq).1).sum();
        let chars: usize = seqs.iter().map(|seq| seq.len() - 1).sum();
        loss / chars as f32
    }
}

/// Training parameters, as sent by the page. Missing fields take defaults.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Params {
    pub hidden: usize,
    pub lr: f32,
    pub clip: f32,
    pub epochs: usize,
    /// Record a point every this many training names.
    pub log_every: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params { hidden: 100, lr: 0.003, clip: 50.0, epochs: 5, log_every: 2000 }
    }
}

impl Params {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.hidden) {
            return Err(format!("hidden must be 1..=1000, got {}", self.hidden));
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(format!("lr must be a positive number, got {}", self.lr));
        }
        if !(self.clip.is_finite() && self.clip > 0.0) {
            return Err(format!("clip must be a positive number, got {}", self.clip));
        }
        if !(1..=100).contains(&self.epochs) {
            return Err(format!("epochs must be 1..=100, got {}", self.epochs));
        }
        if self.log_every == 0 {
            return Err("log_every must be at least 1".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Status {
    Running,
    /// Stopped by the user.
    Stopped,
    /// Finished all epochs.
    Done,
    /// The loss became NaN or infinite, so training stopped by itself.
    Diverged,
}

/// Progress recorded every `log_every` names (and once more at the end).
#[derive(Clone, Debug, Serialize)]
pub struct Point {
    pub names_seen: usize,
    pub epoch: usize,
    /// Mean train loss per character over the interval since the last point.
    pub train_loss: f32,
    /// Mean loss per character on the first `VAL_SUBSET` validation names.
    pub val_loss: f32,
    pub grad_norm_mean: f32,
    pub grad_norm_max: f32,
    /// Share of steps in the interval whose gradient norm exceeded `clip`.
    pub clipped_fraction: f32,
}

#[derive(Serialize)]
pub struct Run {
    pub id: usize,
    pub params: Params,
    pub status: Status,
    pub points: Vec<Point>,
    /// Names generated at the latest point.
    pub samples: Vec<String>,
    /// Weights at the latest point; used for generation while training goes on.
    #[serde(skip)]
    pub rnn: Rnn,
}

#[derive(Debug, PartialEq)]
pub enum StartError {
    Invalid(String),
    /// Another run is still training.
    Busy,
}

#[derive(Debug, PartialEq)]
pub enum ProbeError {
    UnknownRun,
    UnknownChar(char),
    PrefixTooLong,
    BadTemperature,
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProbeError::UnknownRun => write!(f, "no such run"),
            ProbeError::UnknownChar(c) => write!(f, "'{c}' is not in the alphabet"),
            ProbeError::PrefixTooLong => write!(f, "prefix is longer than {MAX_NAME_LEN} letters"),
            ProbeError::BadTemperature => write!(f, "temperature must be a positive number"),
        }
    }
}

/// All runs of this server session. Shared as `Arc<Mutex<Lab>>` between the
/// HTTP handlers and the training thread.
pub struct Lab {
    pub data: Arc<Dataset>,
    pub runs: Vec<Run>,
    /// The run that is training now, with its stop flag.
    active: Option<(usize, Arc<AtomicBool>)>,
}

impl Lab {
    pub fn new(data: Dataset) -> Self {
        Lab { data: Arc::new(data), runs: Vec::new(), active: None }
    }

    pub fn is_training(&self) -> bool {
        self.active.is_some()
    }

    /// Asks run `id` to stop after its current name. Stopping a run that is
    /// not training does nothing.
    pub fn stop(&self, id: usize) -> Result<(), ProbeError> {
        self.runs.get(id).ok_or(ProbeError::UnknownRun)?;
        if let Some((active, flag)) = &self.active
            && *active == id
        {
            flag.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn run(&self, id: usize) -> Result<&Run, ProbeError> {
        self.runs.get(id).ok_or(ProbeError::UnknownRun)
    }

    /// `n` names from run `id`'s latest weights, each starting with `prefix`.
    pub fn generate(
        &self,
        id: usize,
        prefix: &str,
        temperature: f32,
        n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<String>, ProbeError> {
        let run = self.run(id)?;
        check_temperature(temperature)?;
        let prefix = self.data.encode_prefix(prefix)?;
        let (start, end) = (self.data.start(), self.data.end());
        Ok((0..n)
            .map(|_| self.data.decode(&run.rnn.sample(start, end, &prefix, MAX_NAME_LEN, temperature, rng)))
            .collect())
    }

    /// The `TOP_PROBS` most likely next letters after `prefix`, most likely
    /// first. `END` (shown as '>') means "the name ends here".
    pub fn top_probs(&self, id: usize, prefix: &str, temperature: f32) -> Result<Vec<(char, f32)>, ProbeError> {
        let run = self.run(id)?;
        check_temperature(temperature)?;
        let prefix = self.data.encode_prefix(prefix)?;
        let start = self.data.start();
        let mut probs: Vec<(char, f32)> = run
            .rnn
            .next_probs(start, &prefix, temperature)
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != start)
            .map(|(i, p)| (self.data.alphabet[i], p))
            .collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        probs.truncate(TOP_PROBS);
        Ok(probs)
    }
}

fn check_temperature(t: f32) -> Result<(), ProbeError> {
    if t.is_finite() && t > 0.0 { Ok(()) } else { Err(ProbeError::BadTemperature) }
}

/// Validates `params`, adds a new run and trains it on a background thread.
/// Returns the new run's id.
pub fn start_run(lab: &Arc<Mutex<Lab>>, params: Params) -> Result<usize, StartError> {
    params.validate().map_err(StartError::Invalid)?;
    let mut guard = lab.lock().unwrap();
    if guard.active.is_some() {
        return Err(StartError::Busy);
    }
    let id = guard.runs.len();
    let mut rng = StdRng::seed_from_u64(id as u64);
    let rnn = Rnn::new(guard.data.alphabet.len(), params.hidden, &mut rng);
    guard.runs.push(Run {
        id,
        params: params.clone(),
        status: Status::Running,
        points: Vec::new(),
        samples: Vec::new(),
        rnn: rnn.clone(),
    });
    let stop = Arc::new(AtomicBool::new(false));
    guard.active = Some((id, Arc::clone(&stop)));
    let data = Arc::clone(&guard.data);
    drop(guard);

    let lab = Arc::clone(lab);
    std::thread::spawn(move || {
        let status = train(&lab, &data, id, &params, rnn, rng, &stop);
        let mut guard = lab.lock().unwrap();
        guard.runs[id].status = status;
        guard.active = None;
    });
    Ok(id)
}

/// Sums over the names since the last point.
#[derive(Default)]
struct Interval {
    names: usize,
    chars: usize,
    loss_sum: f32,
    norm_sum: f32,
    norm_max: f32,
    clipped: usize,
}

/// The training loop of one run. Returns how it ended.
fn train(
    lab: &Mutex<Lab>,
    data: &Dataset,
    id: usize,
    params: &Params,
    mut rnn: Rnn,
    mut rng: StdRng,
    stop: &AtomicBool,
) -> Status {
    let mut train = data.train.clone();
    let val = &data.val[..VAL_SUBSET.min(data.val.len())];
    let mut interval = Interval::default();
    let mut names_seen = 0;
    let mut epoch = 1;
    let mut status = Status::Done;

    'epochs: while epoch <= params.epochs {
        train.shuffle(&mut rng);
        for seq in &train {
            if stop.load(Ordering::Relaxed) {
                status = Status::Stopped;
                break 'epochs;
            }
            let (loss, dwxh, dwhh, dwhy, dbh, dby, _h) = rnn.train_step(seq);
            let norm = rnn.sgd_update(params.lr, params.clip, [&dwxh, &dwhh, &dwhy, &dbh, &dby]);
            interval.names += 1;
            interval.chars += seq.len() - 1;
            interval.loss_sum += loss;
            interval.norm_sum += norm;
            interval.norm_max = interval.norm_max.max(norm);
            interval.clipped += (norm > params.clip) as usize;
            names_seen += 1;
            if !loss.is_finite() {
                status = Status::Diverged;
                break 'epochs;
            }
            if names_seen % params.log_every == 0 {
                publish(lab, data, id, &rnn, val, names_seen, epoch, &interval, &mut rng);
                interval = Interval::default();
            }
        }
        epoch += 1;
    }
    if interval.names > 0 {
        publish(lab, data, id, &rnn, val, names_seen, epoch.min(params.epochs), &interval, &mut rng);
    }
    status
}

/// Computes a point (outside the lock) and stores it with fresh samples and
/// a snapshot of the weights in run `id`.
#[allow(clippy::too_many_arguments)]
fn publish(
    lab: &Mutex<Lab>,
    data: &Dataset,
    id: usize,
    rnn: &Rnn,
    val: &[Vec<usize>],
    names_seen: usize,
    epoch: usize,
    interval: &Interval,
    rng: &mut StdRng,
) {
    let n = interval.names as f32;
    let point = Point {
        names_seen,
        epoch,
        train_loss: interval.loss_sum / interval.chars as f32,
        val_loss: data.mean_loss(rnn, val),
        grad_norm_mean: interval.norm_sum / n,
        grad_norm_max: interval.norm_max,
        clipped_fraction: interval.clipped as f32 / n,
    };
    let samples = (0..N_SAMPLES)
        .map(|_| data.decode(&rnn.sample(data.start(), data.end(), &[], MAX_NAME_LEN, 1.0, rng)))
        .collect();
    let mut guard = lab.lock().unwrap();
    let run = &mut guard.runs[id];
    run.points.push(point);
    run.samples = samples;
    run.rnn = rnn.clone();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Alphabet `<>ab`; every name is "ab".
    fn toy_data() -> Dataset {
        let alphabet = vec![START, END, 'a', 'b'];
        let char2ind = alphabet.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let name = vec![0, 2, 3, 1];
        Dataset { alphabet, char2ind, train: vec![name.clone(); 50], val: vec![name; 5] }
    }

    fn toy_lab() -> Arc<Mutex<Lab>> {
        Arc::new(Mutex::new(Lab::new(toy_data())))
    }

    fn params(epochs: usize, log_every: usize) -> Params {
        Params { hidden: 8, lr: 0.1, clip: 5.0, epochs, log_every }
    }

    /// Waits until no run is training; panics after 10 s.
    fn wait_idle(lab: &Arc<Mutex<Lab>>) {
        let t0 = Instant::now();
        while lab.lock().unwrap().is_training() {
            assert!(t0.elapsed() < Duration::from_secs(10), "run did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn run_records_points_and_finishes() {
        let lab = toy_lab();
        let id = start_run(&lab, params(2, 10)).unwrap();
        wait_idle(&lab);
        let guard = lab.lock().unwrap();
        let run = &guard.runs[id];
        assert_eq!(run.status, Status::Done);
        assert_eq!(run.points.len(), 10); // 2 epochs × 50 names / 10
        assert_eq!(run.points.last().unwrap().names_seen, 100);
        assert_eq!(run.points.last().unwrap().epoch, 2);
        assert_eq!(run.samples.len(), N_SAMPLES);
        let (first, last) = (&run.points[0], run.points.last().unwrap());
        assert!(last.train_loss < first.train_loss, "loss did not go down");
    }

    #[test]
    fn partial_last_interval_is_recorded() {
        let lab = toy_lab();
        let id = start_run(&lab, params(1, 30)).unwrap(); // 50 names: points at 30 and 50
        wait_idle(&lab);
        let seen: Vec<usize> = lab.lock().unwrap().runs[id].points.iter().map(|p| p.names_seen).collect();
        assert_eq!(seen, vec![30, 50]);
    }

    #[test]
    fn second_run_while_training_is_refused_and_stop_works() {
        let lab = toy_lab();
        let id = start_run(&lab, params(100, 10)).unwrap();
        assert_eq!(start_run(&lab, params(1, 10)), Err(StartError::Busy));
        lab.lock().unwrap().stop(id).unwrap();
        wait_idle(&lab);
        assert_eq!(lab.lock().unwrap().runs[id].status, Status::Stopped);
        assert_eq!(start_run(&lab, params(1, 10)), Ok(id + 1));
        wait_idle(&lab);
    }

    #[test]
    fn invalid_params_are_refused() {
        let lab = toy_lab();
        for bad in [
            Params { lr: 0.0, ..params(1, 10) },
            Params { lr: f32::NAN, ..params(1, 10) },
            Params { hidden: 0, ..params(1, 10) },
            Params { log_every: 0, ..params(1, 10) },
            Params { epochs: 0, ..params(1, 10) },
            Params { clip: -1.0, ..params(1, 10) },
        ] {
            assert!(matches!(start_run(&lab, bad), Err(StartError::Invalid(_))));
        }
        assert!(lab.lock().unwrap().runs.is_empty());
    }

    #[test]
    fn huge_learning_rate_diverges_instead_of_hanging() {
        let lab = toy_lab();
        let id = start_run(&lab, Params { lr: 1e30, clip: 1e30, ..params(100, 10) }).unwrap();
        wait_idle(&lab);
        assert_eq!(lab.lock().unwrap().runs[id].status, Status::Diverged);
        // The page must still be able to serialize the run (NaN becomes null).
        serde_json::to_string(&lab.lock().unwrap().runs).unwrap();
    }

    #[test]
    fn probes_check_their_input() {
        let lab = toy_lab();
        let id = start_run(&lab, params(1, 10)).unwrap();
        wait_idle(&lab);
        let guard = lab.lock().unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        let names = guard.generate(id, "a", 1.0, 4, &mut rng).unwrap();
        assert_eq!(names.len(), 4);
        assert!(names.iter().all(|n| n.starts_with('a')));

        let top = guard.top_probs(id, "a", 1.0).unwrap();
        assert_eq!(top.len(), 3); // 'b', 'a', '>' — START is never offered
        assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));

        assert_eq!(guard.generate(id + 1, "", 1.0, 1, &mut rng), Err(ProbeError::UnknownRun));
        assert_eq!(guard.top_probs(id, "ax", 1.0), Err(ProbeError::UnknownChar('x')));
        assert_eq!(guard.top_probs(id, "a<", 1.0), Err(ProbeError::UnknownChar('<')));
        assert_eq!(guard.top_probs(id, "a", 0.0), Err(ProbeError::BadTemperature));
        assert_eq!(guard.top_probs(id, &"a".repeat(MAX_NAME_LEN + 1), 1.0), Err(ProbeError::PrefixTooLong));
        assert_eq!(guard.stop(id + 1), Err(ProbeError::UnknownRun));
    }
}
