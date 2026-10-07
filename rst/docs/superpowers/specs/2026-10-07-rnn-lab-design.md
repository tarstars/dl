# RNN Lab: a local web app for training and probing the place-name RNN

## Purpose

A learning tool. The user sets training parameters in a browser, starts a
run, watches the loss and gradient curves live, compares runs, and probes the
trained model: generate names from a custom beginning, change the sampling
temperature, and see the next-letter probability distribution.

The user wants to concentrate on DL concepts, so Claude writes the web
plumbing. The DL additions (gradient norm, temperature, prefix, next-letter
probabilities) go into `model.rs`, where the user can study them.

## Scope

In:
- Training parameters: hidden size, learning rate, gradient clip, epochs, log interval.
- Start and stop a run; only one run trains at a time.
- Live charts: train and validation loss; gradient norm with the clip threshold.
- Overlay of all runs of the current server session on the same charts.
- Generation from a prefix, with a temperature slider and a count of names.
- Top-10 next-letter probabilities for the current prefix and temperature,
  updated as the user types.

Out (YAGNI):
- Persistence: runs live in memory and are lost when the server stops.
- Several runs training at once, users, authentication, remote access.
- A frontend build step or framework.

## Architecture

### Crate layout

- `src/lib.rs` (new): declares `pub mod data; pub mod model;` and the shared
  `Result` alias that `data.rs` currently takes from `main.rs`.
- `src/main.rs`: the existing CLI trainer, now using the library crate.
  Behaviour unchanged.
- `src/bin/server.rs` (new): the web app. Run with
  `cargo run --release --bin server`, then open http://127.0.0.1:3000.
- `src/lab.rs` (new, in the library): training-run logic that does not
  depend on HTTP (the `Run` type, the training loop that reports progress).
- `static/index.html` (new): the whole page (HTML, CSS, JS) in one file,
  embedded in the binary with `include_str!`. Chart.js is loaded from
  cdnjs, so the browser needs internet access.

New dependencies: `axum`, `tokio` (features `rt-multi-thread`, `macros`).
`serde` gains `Serialize` derives for API types.

### Model changes (`model.rs`)

1. `#[derive(Clone)]` on `Rnn`, so the trainer can hand out weight snapshots.
2. **Global-norm gradient clipping** replaces per-element clipping:
   `sgd_update` computes `‖g‖ = sqrt(Σ over all five gradients of g²)`; if it
   exceeds `clip`, every gradient is scaled by `clip / ‖g‖`. It returns the
   norm before clipping. This makes the clip threshold a line on the
   gradient-norm chart, and it is the standard method in practice.
3. `fn probs(&self, h, tc, temperature) -> (h_next, p)`: one step that
   returns `softmax(y / T)`. Used by both functions below.
4. `pub fn next_probs(&self, prefix: &[usize], temperature: f32) -> Vec<f32>`:
   feeds `START` + prefix, returns the distribution of the next letter.
5. `pub fn sample(&self, prefix, start, end, max_len, temperature, rng)`:
   feeds `START` + prefix, then draws letters as now. The result includes
   the prefix.
6. The CLI `main.rs` is updated to the new `sample` and `sgd_update`
   signatures (temperature 1.0, empty prefix).

### Lab state (`lab.rs` and `server.rs`)

```
Lab (Arc<Mutex<_>>)
├── alphabet, char2ind, train, val   loaded once at start-up
├── runs: Vec<Run>
└── active: Option<(run id, Arc<AtomicBool> stop flag)>

Run
├── id, params, status (Running | Stopped | Done)
├── points: Vec<Point>     one per log interval
├── samples: Vec<String>   3 names at the latest point
└── rnn: Rnn               weights snapshot at the latest point

Point
├── names_seen
├── train_loss             mean per predicted char over the interval
├── val_loss               on a fixed 1,000-name validation subset
├── grad_norm_mean, grad_norm_max   over the interval
└── clipped_fraction       share of steps where the norm exceeded clip
```

The training thread is a plain `std::thread`. It owns its own `Rnn` and a
copy of the training set, and locks the shared state only at each log point
to push a `Point`, the samples and a weight snapshot. It checks the stop flag
after every name. Its random generator is seeded from the run id, so a run
is reproducible.

### HTTP API

| Method | Path | Does |
|---|---|---|
| GET | `/` | the page |
| POST | `/api/runs` | start a run; body is the params JSON; 409 if a run is active |
| POST | `/api/runs/{id}/stop` | set the stop flag |
| GET | `/api/runs` | all runs: params, status, points, samples (no weights) |
| GET | `/api/runs/{id}/generate?prefix=&temperature=&n=` | `n` names |
| GET | `/api/runs/{id}/probs?prefix=&temperature=` | top 10 `(letter, p)` |

Defaults: hidden 100, lr 0.003, clip 5.0, epochs 5, log interval 2,000
names, temperature 1.0, n 10.

Errors return a status code and a plain-text message, which the page shows:
unknown run id (404), a prefix character that is not in the alphabet (400,
naming the character), invalid params such as `lr <= 0` or `hidden == 0`
(400).

### Page

- **Left column:** the parameter form and Start/Stop buttons; the run list
  with status and latest losses; a run selector.
- **Charts:**
  - Loss: train (solid) and validation (dashed) against names seen, one
    colour per run.
  - Gradient norm: mean and max for the selected run, with the clip level
    as a horizontal line.
- **Probe panel:** a prefix box, a temperature slider (0.1–3), a "Generate"
  button and the list of names, plus top-10 next-letter bars that refresh
  as the prefix or temperature changes.
- Polls `GET /api/runs` every 500 ms while a run is active, otherwise
  only after user actions.

## Testing

`cargo test` unit tests in `model.rs`:
- **Gradient check:** on a tiny model, the analytic gradients from
  `train_step` match central differences `(L(w+ε) − L(w−ε)) / 2ε` within
  tolerance. This is also a teaching artifact.
- `next_probs` sums to 1; low temperature puts almost all mass on the argmax.
- `sample` output starts with the prefix and never contains `START` or `END`.
- Global-norm clipping: the clipped gradient norm equals `clip` when it was
  above it, and the gradients are unchanged when it was below.

The server is checked by hand: start a run, watch the curves, stop it,
start a second run with a different lr, compare, generate.
