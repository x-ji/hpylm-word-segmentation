use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const BOS_CHAR: char = 'Α';
pub const EOS_CHAR: char = 'Ω';
// It's awkward but Rust doesn't support calculating a const yet...
pub const BOS: u64 = 3902801122696021506;
pub const EOS: u64 = 13786557304133791402;
pub const HPYLM_INITIAL_D: f64 = 0.5;
pub const HPYLM_INITIAL_THETA: f64 = 2.0;
pub const BOW: char = 'α';
pub const EOW: char = 'ω';
pub const HPYLM_A: f64 = 1.0;
pub const HPYLM_B: f64 = 1.0;
pub const HPYLM_ALPHA: f64 = 1.0;
pub const HPYLM_BETA: f64 = 1.0;
pub const CHPYLM_BETA_STOP: f64 = 0.57;
pub const CHPYLM_BETA_PASS: f64 = 0.85;
pub const CHPYLM_EPSILON: f64 = 1e-12;
pub const INITIAL_LAMBDA_A: f64 = 4.0;
pub const INITIAL_LAMBDA_B: f64 = 1.0;

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
