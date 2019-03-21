extern crate either;
extern crate rand;
extern crate statrs;

mod sentence;
pub use sentence::Sentence;

mod def;

mod ctype;
mod wtype;

mod corpus;

mod chpylm;
mod hpylm;
mod npylm;
mod whpylm;

mod pyp;
