use corpus::*;
use def::*;
use model::*;
use rand::distributions::Gamma;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;
use wtype::*;

pub struct Trainer {
    rand_indices_train: Vec<usize>,
    rand_indices_dev: Vec<usize>,
    dataset: Dataset,
    // vocabulary: Vocabulary,
    model: Model,
    chpylm_sampling_probability_table: Vec<f64>,
    chpylm_sampling_id_table: Vec<char>,
    always_accept_new_segmentation: bool,
    added_to_chpylm_train: Vec<bool>,
    num_segmentation_rejections: usize,
    num_segmentation_acceptances: usize,
}

impl Trainer {
    pub fn new(dataset: Dataset, model: Model, always_accept_new_segmentation: bool) -> Self {
        let mut rand_indices_train = vec![0; dataset.train_sentences.len()];
        for i in 0..dataset.train_sentences.len() {
            rand_indices_train[i] = i;
        }
        let mut rand_indices_dev = vec![0; dataset.dev_sentences.len()];
        for i in 0..dataset.dev_sentences.len() {
            rand_indices_dev[i] = i;
        }

        Self {
            model: model,
            chpylm_sampling_probability_table: vec![
                0.0;
                dataset.vocabulary.get_num_characters() + 1
            ],
            chpylm_sampling_id_table: vec![' '; dataset.vocabulary.get_num_characters() + 1],
            added_to_chpylm_train: vec![false; dataset.train_sentences.len()],
            dataset: dataset,
            rand_indices_train: rand_indices_train,
            rand_indices_dev: rand_indices_dev,
            always_accept_new_segmentation: always_accept_new_segmentation,
            num_segmentation_acceptances: 0,
            num_segmentation_rejections: 0,
        }
    }

    pub fn sample_hyperparameters(&mut self) {
        self.model.sampler.npylm.sample_hyperparameters();
    }

    pub fn sample_lambda(&mut self) {
        let mut a_array = vec![self.model.sampler.npylm.lambda_a; WORDTYPE_NUM_TYPES];
        let mut b_array = vec![self.model.sampler.npylm.lambda_b; WORDTYPE_NUM_TYPES];
        let mut word_ids: HashSet<u64> = HashSet::new();
        // This method of storing the dataset is hugely problematic. Surely we've got some better ways then. Let's go on of course go on.
        for sentence in &self.dataset.train_sentences {
            for index in 2..sentence.num_segments - 1 {
                let word = sentence.get_nth_word_string(index);
                let word_id = sentence.get_nth_word_id(index);
                let word_length = sentence.get_nth_segment_length(index);
                if word_length > self.model.sampler.npylm.max_word_length {
                    continue;
                }

                if !word_ids.contains(&word_id) {
                    let tablegroups = self
                        .model
                        .sampler
                        .npylm
                        .whpylm
                        .root
                        .tablegroups
                        .get(&word_id)
                        // This word should always be present. Otehrwise it would be a bug.
                        .unwrap();
                    // .unwrap_or(&Vec::new());
                    let num_tablegroups = tablegroups.len();
                    let t = detect_word_type(&word);
                    a_array[t] += (num_tablegroups * word_length) as f64;
                    b_array[t] += num_tablegroups as f64;
                    word_ids.insert(word_id);
                }
            }
            for t in 1..WORDTYPE_NUM_TYPES + 1 {
                let dist = Gamma::new(a_array[t], 1.0 / b_array[t]);
                self.model.sampler.npylm.lambda_for_types[t] = dist.sample(&mut thread_rng());
            }
        }
    }

    fn sample_next_char_from_chpylm_given_context(
        &mut self,
        context_chars: &Vec<char>,
        context_length: usize,
        // This is apparently unused
        _sample_t: usize,
        skip_eow: bool,
    ) -> char {
        // let mut prob_sum = 0.0;
        let mut table_index = 1;
        // let num_characters = self.dataset.vocabulary.all_characters.len();
        for c in &self.dataset.vocabulary.all_characters {
            let p_w = self
                .model
                .sampler
                .npylm
                .chpylm
                .compute_p_w_given_h_with_target(*c, context_chars, 0, context_length - 1);

            // prob_sum += p_w;
            self.chpylm_sampling_probability_table[table_index] = p_w;
            self.chpylm_sampling_id_table[table_index] = *c;
            table_index += 1;
        }

        if !skip_eow {
            let p_w = self
                .model
                .sampler
                .npylm
                .chpylm
                .compute_p_w_given_h_with_target(EOW, context_chars, 0, context_length - 1);
            // prob_sum += p_w;
            self.chpylm_sampling_probability_table[table_index] = p_w;
            self.chpylm_sampling_id_table[table_index] = EOW;
        }

        let dist = WeightedIndex::new(&self.chpylm_sampling_probability_table).unwrap();
        let index_of_char = dist.sample(&mut rand::thread_rng());
        return self.chpylm_sampling_id_table[index_of_char];
    }

    pub fn update_p_k_given_chpylm_default(&mut self) {
        self.update_p_k_given_chpylm(20000, 10);
    }

    pub fn update_p_k_given_chpylm(&mut self, num_samples: usize, early_stopping_threshold: usize) {
        let max_word_length = self.model.get_max_word_length() + 1;
        for i in 0..max_word_length + 1 {
            self.model.sampler.npylm.p_k_chpylm[i] = 0.0;
        }
        let mut num_words_of_length_k = vec![0; max_word_length + 1];

        let mut wrapped_chars = vec![' '; max_word_length + 3];
        let mut num_words_sampled = 0;

        for itr in 1..num_samples {
            wrapped_chars[0] = BOW;

            let mut cur_word_length = 0;
            for j in 0..max_word_length {
                let skip_eow = if j == 0 { true } else { false };
                let next_char = self.sample_next_char_from_chpylm_given_context(
                    &mut wrapped_chars,
                    j + 1,
                    j + 1,
                    skip_eow,
                );
                wrapped_chars[j + 1] = next_char;
                if next_char == EOW {
                    break;
                }
                cur_word_length += 1;
            }

            num_words_sampled += 1;

            if cur_word_length == 0 {
                continue;
            }

            num_words_of_length_k[cur_word_length] += 1;

            if itr % 100 == 0 {
                let mut can_stop = true;
                for k in 1..max_word_length + 1 {
                    if num_words_of_length_k[k] < early_stopping_threshold {
                        can_stop = false;
                        break;
                    }
                }
                if can_stop {
                    break;
                }
            }
        }

        for k in 1..max_word_length + 1 {
            self.model.sampler.npylm.p_k_chpylm[k] = (num_words_of_length_k[k] + 1) as f64
                / (num_words_sampled + max_word_length) as f64;

            assert!(self.model.sampler.npylm.p_k_chpylm[k] > 0.0);
        }
    }

    pub fn blocked_gibbs_sampling(&mut self) {
        let num_sentences = self.dataset.train_sentences.len();
        let max_sentence_length = self.dataset.max_sentence_length;

        self.rand_indices_train.shuffle(&mut thread_rng());

        for step in 1..num_sentences + 1 {
            let sentence_index = self.rand_indices_train[step - 1];
            let sentence = &mut self.dataset.train_sentences[sentence_index];

            if sentence.supervised {
                if self.added_to_chpylm_train[sentence_index] == true {
                    for n in 2..sentence.num_segments {
                        self.model
                            .sampler
                            .npylm
                            .remove_customer_at_index_n(sentence, n);
                    }
                }

                for n in 2..sentence.num_segments {
                    self.model
                        .sampler
                        .npylm
                        .add_customer_at_index_n(sentence, n);
                }

                self.added_to_chpylm_train[sentence_index] = true;
            } else {
                if self.added_to_chpylm_train[sentence_index] == true {
                    let mut old_segment_lengths = vec![0; max_sentence_length + 3];
                    let mut num_old_segments = 0;
                    let mut old_log_p_s = 0.0;

                    for n in 2..sentence.num_segments {
                        self.model
                            .sampler
                            .npylm
                            .remove_customer_at_index_n(sentence, n);
                    }

                    if !self.always_accept_new_segmentation {
                        num_old_segments = sentence.get_num_segments_without_special_tokens();
                        for i in 0..num_old_segments {
                            old_segment_lengths[i] = sentence.segment_lengths[i + 2];
                        }
                        old_log_p_s = self
                            .model
                            .sampler
                            .npylm
                            .compute_log_probability_of_sentence(sentence);
                    }

                    let new_segment_lengths =
                        self.model.sampler.blocked_gibbs_segment(sentence, true);

                    sentence.split_sentence(new_segment_lengths);

                    if !self.always_accept_new_segmentation {
                        let new_log_p_s = self
                            .model
                            .sampler
                            .npylm
                            .compute_log_probability_of_sentence(sentence);
                        let bernoulli = (1.0 as f64).min((new_log_p_s - old_log_p_s).exp());
                        let r = rand::thread_rng().gen();
                        if bernoulli < r {
                            sentence.split_sentence_with_num_segments(
                                old_segment_lengths,
                                num_old_segments,
                            );
                            self.num_segmentation_rejections += 1;
                        } else {
                            self.num_segmentation_acceptances += 1;
                        }
                    }
                }

                for n in 2..sentence.num_segments {
                    self.model
                        .sampler
                        .npylm
                        .add_customer_at_index_n(sentence, n);
                }
                self.added_to_chpylm_train[sentence_index] = true;
            }
        }
    }

    // pub fn compute_perplexity(&mut self, sentences: &Vec<Sentence>) -> f64 {
    pub fn compute_perplexity(&mut self, train_sentences: bool) -> f64 {
        let sentences = if train_sentences {
            &self.dataset.train_sentences
        } else {
            &self.dataset.dev_sentences
        };

        let num_sentences = sentences.len();

        if num_sentences == 0 {
            return 0.0;
        }

        let mut sum = 0.0;

        for s in sentences {
            let mut sentence = s.clone();
            let segment_lengths = self.model.sampler.viterbi_decode(&sentence);
            sentence.split_sentence(segment_lengths);
            sum += self
                .model
                .sampler
                .npylm
                .compute_log_probability_of_sentence(&sentence)
                / (sentence.num_segments - 2) as f64;
        }

        let ppl = (-sum / num_sentences as f64).exp();
        ppl
    }

    fn compute_perplexity_train(&mut self) -> f64 {
        self.compute_perplexity(true)
    }

    fn compute_perplexity_dev(&mut self) -> f64 {
        self.compute_perplexity(false)
    }

    fn compute_log_likelihood(&mut self, train_sentences: bool) -> f64 {
        let sentences = if train_sentences {
            &self.dataset.train_sentences
        } else {
            &self.dataset.dev_sentences
        };

        let num_sentences = sentences.len();

        if num_sentences == 0 {
            return 0.0;
        }

        let mut sum = 0.0;

        for sentence in sentences {
            let log_p_x = self
                .model
                .sampler
                .compute_log_forward_probability(sentence, true);
            sum += log_p_x;
        }

        sum
    }

    fn compute_log_likelihood_train(&mut self) -> f64 {
        self.compute_log_likelihood(true)
    }

    fn compute_log_likelihood_dev(&mut self) -> f64 {
        self.compute_log_likelihood(false)
    }

    fn print_segmentations(&mut self, num_to_print: usize, train_sentences: bool) {
        let sentences = if train_sentences {
            &self.dataset.train_sentences
        } else {
            &self.dataset.dev_sentences
        };

        let rand_indices = if train_sentences {
            &self.rand_indices_train
        } else {
            &self.rand_indices_dev
        };

        let num_to_print = num_to_print.min(sentences.len());

        for n in 1..num_to_print + 1 {
            let sentence_index = rand_indices[n];
            let mut sentence = sentences[sentence_index].clone();
            let segment_lengths = self.model.sampler.viterbi_decode(&sentence);
            sentence.split_sentence(segment_lengths);
            println!("{}\n", sentence);
        }
    }

    fn print_segmentations_train(&mut self, num_to_print: usize) {
        self.print_segmentations(num_to_print, true);
    }

    fn print_segmentations_dev(&mut self, num_to_print: usize) {
        self.print_segmentations(num_to_print, false);
    }
}
