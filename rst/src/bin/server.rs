//! RNN Lab: a local web app to train the place-name RNN and probe it.
//!
//! Run with `cargo run --release --bin server`, then open http://127.0.0.1:3000.
//! The page is `static/index.html`, compiled into the binary.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rnn_training::lab::{self, Dataset, Lab, Params, ProbeError, StartError};
use serde::{Deserialize, Serialize};
use serde_json::json;

const ADDR: &str = "127.0.0.1:3000";
const DATA_DIR: &str = "/home/tarstars/database/shad/generate_text/ru";
const SEED: u64 = 42;
const MIN_CHAR_COUNT: usize = 10;
const VAL_FRACTION: f64 = 0.1;
/// At most this many names per generate request.
const MAX_GENERATE: usize = 50;

type Shared = Arc<Mutex<Lab>>;

#[tokio::main]
async fn main() -> rnn_training::Result<()> {
    println!("loading names from {DATA_DIR} ...");
    let data = Dataset::load(DATA_DIR, MIN_CHAR_COUNT, VAL_FRACTION, SEED)?;
    println!("train: {} val: {} alphabet: {}", data.train.len(), data.val.len(), data.alphabet.len());
    let lab: Shared = Arc::new(Mutex::new(Lab::new(data)));

    let app = Router::new()
        .route("/", get(index))
        .route("/api/info", get(info))
        .route("/api/runs", get(list_runs).post(start_run))
        .route("/api/runs/{id}/stop", post(stop_run))
        .route("/api/runs/{id}/generate", get(generate))
        .route("/api/runs/{id}/probs", get(probs))
        .with_state(lab);

    let listener = tokio::net::TcpListener::bind(ADDR).await?;
    println!("RNN Lab at http://{ADDR}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

/// Facts about the data that the page shows, and the form defaults.
#[derive(Serialize)]
struct Info {
    alphabet: String,
    /// Loss of a model that guesses every letter equally: ln(alphabet size).
    uniform_loss: f32,
    train_names: usize,
    val_names: usize,
    defaults: Params,
}

async fn info(State(lab): State<Shared>) -> Json<Info> {
    let lab = lab.lock().unwrap();
    let data = &lab.data;
    Json(Info {
        alphabet: data.alphabet.iter().collect(),
        uniform_loss: (data.alphabet.len() as f32).ln(),
        train_names: data.train.len(),
        val_names: data.val.len(),
        defaults: Params::default(),
    })
}

async fn list_runs(State(lab): State<Shared>) -> Response {
    Json(&lab.lock().unwrap().runs).into_response()
}

async fn start_run(State(lab): State<Shared>, Json(params): Json<Params>) -> Response {
    match lab::start_run(&lab, params) {
        Ok(id) => Json(json!({ "id": id })).into_response(),
        Err(StartError::Busy) => (StatusCode::CONFLICT, "a run is already training; stop it first").into_response(),
        Err(StartError::Invalid(msg)) => (StatusCode::BAD_REQUEST, msg).into_response(),
    }
}

async fn stop_run(State(lab): State<Shared>, Path(id): Path<usize>) -> Response {
    match lab.lock().unwrap().stop(id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => probe_error(e),
    }
}

#[derive(Deserialize)]
struct GenerateQuery {
    #[serde(default)]
    prefix: String,
    #[serde(default = "one")]
    temperature: f32,
    #[serde(default = "ten")]
    n: usize,
}

#[derive(Deserialize)]
struct ProbsQuery {
    #[serde(default)]
    prefix: String,
    #[serde(default = "one")]
    temperature: f32,
}

fn one() -> f32 {
    1.0
}

fn ten() -> usize {
    10
}

async fn generate(State(lab): State<Shared>, Path(id): Path<usize>, Query(q): Query<GenerateQuery>) -> Response {
    let mut rng = StdRng::seed_from_u64(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64);
    let n = q.n.min(MAX_GENERATE);
    match lab.lock().unwrap().generate(id, &q.prefix, q.temperature, n, &mut rng) {
        Ok(names) => Json(names).into_response(),
        Err(e) => probe_error(e),
    }
}

async fn probs(State(lab): State<Shared>, Path(id): Path<usize>, Query(q): Query<ProbsQuery>) -> Response {
    match lab.lock().unwrap().top_probs(id, &q.prefix, q.temperature) {
        Ok(top) => Json(top).into_response(),
        Err(e) => probe_error(e),
    }
}

fn probe_error(e: ProbeError) -> Response {
    let status = match e {
        ProbeError::UnknownRun => StatusCode::NOT_FOUND,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, e.to_string()).into_response()
}
