//! Character-level RNN for Russian place names: the data pipeline, the
//! model, and the training runs of the web lab. Used by the CLI trainer
//! (`src/main.rs`) and the web server (`src/bin/server.rs`).

pub mod data;
pub mod model;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
