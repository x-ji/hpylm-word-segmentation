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
pub const HPYLM_INITIAL_d: f64 = 0.5;
pub const HPYLM_INITIAL_theta: f64 = 2.0;
pub const HPYLM_a: f64 = 1.0;
pub const HPYLM_b: f64 = 1.0;
pub const HPYLM_alpha: f64 = 1.0;
pub const HPYLM_beta: f64 = 1.0;
pub const CHPYLM_beta_STOP: f64 = 0.57;
pub const CHPYLM_beta_PASS: f64 = 0.85;
pub const CHPYLM_epsilon: f64 = 1e-12;
pub const INITIAL_LAMBDA_a: f64 = 4.0;
pub const INITIAL_LAMBDA_b: f64 = 1.0;

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
