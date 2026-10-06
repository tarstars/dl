//! Loading and preparing the place-name dataset.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::Result;

/// Marks the start of a name in the RNN input.
pub const START: char = '<';
/// Marks the end of a name in the RNN output.
pub const END: char = '>';

#[derive(Deserialize)]
struct Place {
    name: Option<String>,
}

/// Reads one ndjson file and adds every `name` it contains to `names`.
fn read_names_into(path: &str, names: &mut HashSet<String>) -> Result<()> {
    let file = File::open(path)?;
    for line in BufReader::new(file).lines() {
        let place: Place = serde_json::from_str(&line?)?;
        if let Some(name) = place.name {
            names.insert(name);
        }
    }
    Ok(())
}

/// Unique place names from all files in `dir`, sorted so that the order
/// does not depend on `HashSet` hashing.
pub fn load_names(dir: &str) -> Result<Vec<String>> {
    let mut names = HashSet::new();
    for kind in ["village", "town", "hamlet", "city"] {
        read_names_into(&format!("{dir}/place-{kind}.ndjson"), &mut names)?;
    }
    let mut names: Vec<String> = names.into_iter().collect();
    names.sort();
    Ok(names)
}

/// How many times each character occurs across all names.
pub fn char_counts(names: &[String]) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for name in names {
        for c in name.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
    }
    counts
}

/// Characters that occur fewer than `min_count` times.
pub fn rare_chars(names: &[String], min_count: usize) -> HashSet<char> {
    char_counts(names)
        .into_iter()
        .filter(|&(_, count)| count < min_count)
        .map(|(c, _)| c)
        .collect()
}

/// Keeps only the names that contain none of the `banned` characters.
pub fn drop_names_with(names: Vec<String>, banned: &HashSet<char>) -> Vec<String> {
    names
        .into_iter()
        .filter(|name| !name.chars().any(|c| banned.contains(&c)))
        .collect()
}

/// The sorted characters used in `names`, preceded by `START` and `END`.
pub fn build_alphabet(names: &[String]) -> Vec<char> {
    let letters: HashSet<char> = names.iter().flat_map(|name| name.chars()).collect();
    let mut letters: Vec<char> = letters.into_iter().collect();
    letters.sort();
    [vec![START, END], letters].concat()
}

/// Shuffles `names` and splits off `val_fraction` of them as the validation set.
/// Returns `(train, val)`.
pub fn split_train_val(
    mut names: Vec<String>,
    val_fraction: f64,
    rng: &mut StdRng,
) -> (Vec<String>, Vec<String>) {
    names.shuffle(rng);
    let n_val = (names.len() as f64 * val_fraction) as usize;
    let val = names.split_off(names.len() - n_val);
    (names, val)
}
