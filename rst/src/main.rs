mod data;
mod model;

use model::Rnn;
use rand::SeedableRng;
use rand::rngs::StdRng;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const DATA_DIR: &str = "/home/tarstars/database/shad/generate_text/ru";
const SEED: u64 = 42;
/// Names containing a character seen fewer times than this are dropped.
const MIN_CHAR_COUNT: usize = 10;
const VAL_FRACTION: f64 = 0.1;
const HIDDEN_SIZE: usize = 100;

fn main() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let names = data::load_names(DATA_DIR)?;
    let n_loaded = names.len();
    let rare = data::rare_chars(&names, MIN_CHAR_COUNT);
    let names = data::drop_names_with(names, &rare);
    let alphabet = data::build_alphabet(&names);
    let (train, val) = data::split_train_val(names, VAL_FRACTION, &mut rng);

    println!(
        "names: {n_loaded} loaded, {} dropped for rare characters",
        n_loaded - train.len() - val.len()
    );
    println!("train: {} val: {}", train.len(), val.len());
    println!(
        "alphabet ({}): '{}'",
        alphabet.len(),
        alphabet.iter().collect::<String>()
    );

    let rnn = Rnn::new(alphabet.len(), HIDDEN_SIZE, &mut rng);
    println!("parameters: {}", rnn.n_params());

    Ok(())
}
