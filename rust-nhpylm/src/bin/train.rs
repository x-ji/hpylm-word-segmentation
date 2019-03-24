extern crate either;
extern crate getopts;
extern crate rust_nhpylm;

use either::*;
use std::env::args;
use std::fs;
use std::fs::{read_dir, File};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use getopts::Options;
use std::process;

use rust_nhpylm::def::*;
use rust_nhpylm::{Corpus, Dataset, Model, Trainer};

// Either Left(file) or Right(dir).
fn build_corpus(path: Either<&str, &str>) -> Corpus {
    // fn build_corpus(input_file_name: &str) -> Corpus {
    let mut corpus = Corpus::new();
    match path {
        Left(input_file_name) => {
            let path = Path::new(input_file_name).to_path_buf();
            corpus.read_corpus(&path);
        }
        Right(input_dir_name) => {
            for file in fs::read_dir(input_dir_name).unwrap() {
                let file = file.unwrap().path();
                corpus.read_corpus(&file);
            }
        }
    }
    corpus
}

// fn read_file_into_corpus(path: &str, corpus: &mut Corpus) {
//     let input_file = File::open(path).unwrap();
//     let input_file_reader = &mut BufReader::new(input_file);
//     for line in input_file_reader.lines() {
//         let l = line.unwrap();
//         if l.is_empty() {
//             continue;
//         }

//     }
// }

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "Print this help menu");
    opts.optopt("f", "file", "Path to the training file", "FILENAME");
    opts.optopt("d", "dir", "Path to the training directory", "DIRNAME");

    opts.optflag(
        "a",
        "always-accept-new-segmentation",
        "Always accept new segmentation",
    );

    opts.optopt("s", "seed", "Seed for the training", "1");
    opts.optopt("e", "epochs", "Total epochs of training", "100000");
    // opts.optopt(
    //     "",
    //     "target-directory",
    //     "Directory to save the trained model to",
    //     "out",
    // );
    opts.optopt(
        "p",
        "train-dev-split",
        "The split proportion between training data and dev data",
        "0.9",
    );

    opts.optopt("", "lambda-a", "", "4.0");
    opts.optopt("", "lambda-b", "", "1.0");
    opts.optopt(
        "",
        "beta-stop",
        "The beta-stop parameter for character-level HPYLM",
        "0.57",
    );
    opts.optopt(
        "",
        "beta-pass",
        "The beta-pass parameter for character-level HPYLM",
        "0.85",
    );
    opts.optopt(
        "l",
        "max-word-length",
        "Maximum allowed length of a word",
        "16",
    );

    let matches = opts.parse(&args[1..]).unwrap_or_else(|e| {
        println!("Error: {}", e);
        process::exit(1);
    });

    if !matches.opts_present(&["f".to_owned(), "d".to_owned()]) {
        println!("Please specify either the corpus file with -f or the corpus directory with -d!");
        process::exit(1);
    }

    if matches.opt_present("f") && matches.opt_present("d") {
        println!("Please specify either the corpus file or directory, but not both!");
        process::exit(1);
    }

    // let target_directory = matches.opt_get_default("target-directory", "out".to_owned());

    let seed = matches.opt_get_default("s", 1).unwrap();
    let epoches = matches.opt_get_default("e", 100000).unwrap();
    let split = matches.opt_get_default("p", 0.9).unwrap();
    let lambda_a = matches
        .opt_get_default("lambda-a", INITIAL_LAMBDA_a)
        .unwrap();
    let lambda_b = matches
        .opt_get_default("lambda-b", INITIAL_LAMBDA_b)
        .unwrap();
    let beta_stop = matches
        .opt_get_default("beta-stop", CHPYLM_beta_STOP)
        .unwrap();
    let beta_pass = matches
        .opt_get_default("beta-stop", CHPYLM_beta_PASS)
        .unwrap();
    let max_word_length = matches.opt_get_default("max-word-length", 16).unwrap();

    let always_accept_new_segmentation = if matches.opt_present("a") {
        true
    } else {
        false
    };

    let corpus = if matches.opt_present("f") {
        let input_file: String = matches.opt_get("f").unwrap().unwrap();
        build_corpus(Left(&input_file))
    } else {
        let input_dir: String = matches.opt_get("d").unwrap().unwrap();
        build_corpus(Right(&input_dir))
    };

    let dataset = Dataset::new(corpus, split, seed);
    println!(
        "Number of train sentences {}",
        dataset.get_num_train_sentences()
    );
    println!(
        "Number of train sentences {}",
        dataset.get_num_dev_sentences()
    );

    let mut model = Model::new(&dataset, max_word_length);
    model.set_initial_a(lambda_a);
    model.set_initial_b(lambda_b);
    model.set_chpylm_beta_stop(beta_stop);
    model.set_chpylm_beta_pass(beta_pass);

    let mut trainer = Trainer::new(dataset, model, always_accept_new_segmentation);

    for epoch in 1..epoches + 1 {
        let start_time = SystemTime::now();
        trainer.blocked_gibbs_sampling();
        trainer.sample_hyperparameters();
        trainer.sample_lambda();

        if epoch > 3 {
            trainer.update_p_k_given_chpylm_default();
        }

        let end_time = SystemTime::now();
        let duration = end_time.duration_since(start_time).unwrap();
        println!(
            "Iteration {}. Elapsed time in this iteration: {}ms",
            epoch,
            duration.as_millis()
        );
    }
}
