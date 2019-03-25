# Rust-NHPYLM

(Note: There are still some bugs in this Rust implementation which need to be ironed out before it runs stably.)

To run the program:
1. Install [Rust](https://www.rust-lang.org/tools/install).
2. Run `cargo build --release` under the `rust-nhpylm` folder.
3. Run `./target/release/train -h` to get help for running the training program.
4. Run `./target/release/train -f FILE -l 4` to perform training on the given corpus, with maximum word length set to 4.
