use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use sentence::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

pub struct Vocabulary {
    all_characters: HashSet<char>,
}

impl Vocabulary {
    fn new() -> Self {
        Self {
            all_characters: HashSet::new(),
        }
    }

    pub fn add_character(&mut self, character: char) {
        self.all_characters.insert(character);
    }

    pub fn get_num_characters(&self) -> usize {
        self.all_characters.len()
    }
}

pub struct Corpus {
    sentence_list: Vec<String>,
    segmented_word_list: Vec<Vec<String>>,
}

impl Corpus {
    fn new() -> Self {
        Self {
            sentence_list: Vec::new(),
            segmented_word_list: Vec::new(),
        }
    }

    pub fn add_sentence(&mut self, sentence_string: String) {
        self.sentence_list.push(sentence_string);
    }

    pub fn read_corpus(&mut self, input_file_path: String) {
        let input_file = File::open(input_file_path).unwrap();
        let reader = &mut BufReader::new(input_file);
        // let mut line = String::new();
        for line in reader.lines() {
            let l = line.unwrap();
            if l.is_empty() {
                continue;
            }
            self.add_sentence(l.trim().to_owned());
        }
    }

    pub fn get_num_sentences(&self) -> usize {
        self.sentence_list.len()
    }

    pub fn get_num_already_segmented_sentences(&self) -> usize {
        self.segmented_word_list.len()
    }
}

struct Dataset {
    vocabulary: Vocabulary,
    corpus: Corpus,
    max_sentence_length: usize,
    avg_sentence_length: f64,
    num_segmented_words: usize,
    train_sentences: Vec<Sentence>,
    dev_sentences: Vec<Sentence>,
}

impl Dataset {
    fn new(corpus: Corpus, train_proportion: f64) -> Self {
        let mut corpus_length = 0;
        let num_sentences = corpus.get_num_sentences();
        let mut vocabulary = Vocabulary::new();
        let mut train_sentences: Vec<Sentence> = Vec::new();
        let mut dev_sentences: Vec<Sentence> = Vec::new();
        let mut max_sentence_length = 0;

        let mut sentence_indices = vec![0; num_sentences];
        for i in 0..num_sentences {
            sentence_indices[i] = i;
        }
        sentence_indices.shuffle(&mut thread_rng());

        let train_proportion = (1.0 as f64).min((0.0 as f64).max(train_proportion));
        let num_train_sentences = corpus.get_num_sentences() * train_proportion.floor() as usize;

        for i in 0..num_sentences {
            // It is actually a reference to something stored in the Corpus struct.
            let sentence_string = &corpus.sentence_list[sentence_indices[i]];

            if i < num_train_sentences {
                add_sentence(
                    &mut vocabulary,
                    &mut train_sentences,
                    sentence_string.to_owned(),
                );
            } else {
                add_sentence(
                    &mut vocabulary,
                    &mut dev_sentences,
                    sentence_string.to_owned(),
                );
            }

            if sentence_string.len() > max_sentence_length {
                max_sentence_length = sentence_string.len();
            }
            corpus_length += sentence_string.len();
        }

        let avg_sentence_length = corpus_length as f64 / num_sentences as f64;

        // let num_supervised_sentences = corpus.get_num_already_segmented_sentences();

        // for i in 0..num_supervised_sentences {
        //     words = corpus.segmented_word_list[i];
        //     segment
        // }

        Self {
            vocabulary: vocabulary,
            corpus: corpus,
            max_sentence_length: max_sentence_length,
            avg_sentence_length: avg_sentence_length,
            train_sentences: train_sentences,
            dev_sentences: dev_sentences,
            // Will be 0 if we don't provide any supervised examples to the training.
            num_segmented_words: 0,
        }
    }

    fn get_num_train_sentences(&self) -> usize {
        self.train_sentences.len()
    }

    fn get_num_dev_sentences(&self) -> usize {
        self.dev_sentences.len()
    }

    // This design is bonkers, at least for Rust, the below version taking in each component individually is much cleaner!
    // fn add_sentence(&mut self, sentence_string: String, is_train: bool) {
    //     for c in sentence_string.chars() {
    //         self.vocabulary.add_character(c);
    //     }
    //     let s = Sentence::new(sentence_string, false);
    //     if is_train {
    //         self.train_sentences.push(s);
    //     } else {
    //         self.dev_sentences.push(s);
    //     }
    // }
}

fn add_sentence(
    vocabulary: &mut Vocabulary,
    sentences: &mut Vec<Sentence>,
    sentence_string: String,
) {
    for c in sentence_string.chars() {
        vocabulary.add_character(c);
    }
    let s = Sentence::new(sentence_string, false);
    sentences.push(s);
}
