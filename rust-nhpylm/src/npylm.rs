use chpylm::*;
use def::*;
use either::*;
use pyp::*;
use rand::distributions::{Bernoulli, Beta, Distribution, Gamma, WeightedIndex};
use rand::prelude::*;
use rand::Rng;
use sentence::*;
use statrs::distribution::{Discrete, Poisson};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use whpylm::*;
use wtype::*;

fn produce_word_with_bow_and_eow(
    sentence_as_chars: &Vec<char>,
    word_begin_index: usize,
    word_end_index: usize,
) -> Vec<char> {
    let mut word = vec![' '; word_end_index - word_begin_index + 3];
    word[0] = BOW;
    let mut i = 0;
    while i < (word_end_index - word_begin_index + 1) {
        word[i + 1] = sentence_as_chars[word_begin_index + 1];
        i += 1;
    }
    word[i + 1] = EOW;
    word
}

pub struct NPYLM {
    whpylm: WHPYLM,
    chpylm: CHPYLM,
    recorded_depth_arrays_for_tablegroups_of_token: HashMap<u64, Vec<Vec<usize>>>,
    whpylm_g_0_cache: HashMap<u64, f64>,
    chpylm_g_0_cache: HashMap<usize, f64>,
    lambda_for_types: Vec<f64>,
    p_k_chpylm: Vec<f64>,
    max_word_length: usize,
    max_sentence_length: usize,
    lambda_a: f64,
    lambda_b: f64,
    whpylm_parent_p_w_cache: Vec<f64>,
    most_recent_word: Vec<char>,
}

impl NPYLM {
    fn new(
        max_word_length: usize,
        max_sentence_length: usize,
        g_0: f64,
        initial_lambda_a: f64,
        initial_lambda_b: f64,
        chpylm_beta_stop: f64,
        chpylm_beta_pass: f64,
    ) -> Self {
        let mut npylm = Self {
            // whpylm: Box::new(WHPYLM::new(3)),
            // chpylm: Box::new(CHPYLM::new(
            //     g_0,
            //     max_sentence_length,
            //     chpylm_beta_stop,
            //     chpylm_beta_pass,
            // )),
            whpylm: WHPYLM::new(3),
            chpylm: CHPYLM::new(g_0, max_sentence_length, chpylm_beta_stop, chpylm_beta_pass),
            recorded_depth_arrays_for_tablegroups_of_token: HashMap::new(),
            whpylm_g_0_cache: HashMap::new(),
            chpylm_g_0_cache: HashMap::new(),
            lambda_for_types: vec![0.0; WORDTYPE_NUM_TYPES],
            whpylm_parent_p_w_cache: vec![0.0; 3],
            lambda_a: initial_lambda_a,
            lambda_b: initial_lambda_b,
            max_sentence_length: max_sentence_length,
            max_word_length: max_word_length,
            p_k_chpylm: vec![1.0 / (max_word_length + 2) as f64; max_word_length + 2],
            most_recent_word: Vec::new(),
        };
        npylm.sample_lambda_with_initial_params();
        npylm
    }

    fn sample_lambda_with_initial_params(&mut self) {
        for i in 1..WORDTYPE_NUM_TYPES + 1 {
            let dist = Gamma::new(self.lambda_a, 1.0 / self.lambda_b);
            self.lambda_for_types[i] = dist.sample(&mut thread_rng());
        }
    }

    fn extend_capacity(&mut self, max_sentence_length: usize) {
        if max_sentence_length <= self.max_sentence_length {
            return;
        } else {
            self.allocate_capacity(max_sentence_length);
        }
    }

    fn allocate_capacity(&mut self, max_sentence_length: usize) {
        self.max_sentence_length = max_sentence_length;
        self.most_recent_word = vec![' '; max_sentence_length + 2];
    }

    fn add_customer_at_index_n(&mut self, sentence: &Sentence, n: usize) -> bool {
        let token_n = sentence.get_nth_word_id(n);
        let pyp = self
            .find_node_with_sentence(sentence, n, true, false)
            .unwrap();
        let num_tables_before_addition = self.whpylm.root.ntables;
        let mut index_of_table_added_to_in_root = 0;
        unsafe {
            (*pyp).add_customer(
                token_n,
                Right(&mut self.whpylm_parent_p_w_cache),
                &mut self.whpylm.d_array,
                &mut self.whpylm.theta_array,
                true,
                &mut index_of_table_added_to_in_root,
            );
            let num_tables_after_addition = self.whpylm.root.ntables;
            let word_begin_index = sentence.segment_begin_positions[n];
            let word_end_index = word_begin_index + sentence.segment_lengths[n] - 1;

            if (num_tables_before_addition < num_tables_after_addition) {
                self.whpylm_g_0_cache = HashMap::new();
                if token_n == EOS {
                    self.chpylm.root.add_customer(
                        EOS_CHAR,
                        Left(self.chpylm.g_0),
                        &mut self.chpylm.d_array,
                        &mut self.chpylm.theta_array,
                        true,
                        &mut index_of_table_added_to_in_root,
                    );
                    return true;
                }

                let depth_arrays_for_the_tablegroup = {
                    self.recorded_depth_arrays_for_tablegroups_of_token
                        .entry(token_n)
                        .or_insert(Vec::new())
                };

                let mut recorded_depth_array = vec![0; word_end_index - word_begin_index + 3];
                self.add_word_to_chpylm(
                    &sentence.characters,
                    word_begin_index,
                    word_end_index,
                    &mut recorded_depth_array,
                );
                depth_arrays_for_the_tablegroup.push(recorded_depth_array);
            }
        }
        return true;
    }

    fn add_word_to_chpylm(
        &mut self,
        sentence_as_chars: &Vec<char>,
        word_begin_index: usize,
        word_end_index: usize,
        recorded_depth_array: &mut Vec<usize>,
    ) {
        self.most_recent_word =
            produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index);
        let word_length_with_symbols = word_end_index - word_begin_index + 1 + 2;
        for n in 0..word_length_with_symbols {
            let depth_n = self
                .chpylm
                .sample_depth_at_index_n(&self.most_recent_word, n);
            self.chpylm
                .add_customer_at_index_n(&self.most_recent_word, n, depth_n, true);
            recorded_depth_array[n] = depth_n;
        }
    }

    fn find_node_with_word_ids() {}

    fn find_node_with_sentence(
        &mut self,
        sentence: &Sentence,
        n: usize,
        generate_if_not_found: bool,
        return_middle_node: bool,
    ) -> Option<*mut PYP<u64>> {
        let word_begin_index = sentence.segment_begin_positions[n];
        let word_end_index = word_begin_index + sentence.segment_lengths[n] - 1;
        return self.find_node_with_sentence_as_chars(
            &sentence.characters,
            &sentence.word_ids,
            n,
            word_begin_index,
            word_end_index,
            generate_if_not_found,
            return_middle_node,
        );
    }

    fn find_node_with_sentence_as_chars(
        &mut self,
        sentence_as_chars: &Vec<char>,
        word_ids: &Vec<u64>,
        n: usize,
        word_begin_index: usize,
        word_end_index: usize,
        generate_if_not_found: bool,
        return_middle_node: bool,
    ) -> Option<*mut PYP<u64>> {
        let mut cur_node = &mut self.whpylm.root as *mut PYP<u64>;
        let word_n_id = word_ids[n];
        let mut parent_p_w = self.compute_g_0_of_word_at_index_n(
            sentence_as_chars,
            word_begin_index,
            word_end_index,
            word_n_id,
        );
        self.whpylm_parent_p_w_cache[0] = parent_p_w;
        unsafe {
            for depth in 1..3 {
                let context = BOS;
                if n - depth >= 0 {
                    let context = word_ids[n - depth];
                }
                let p_w = (*cur_node).compute_p_w_with_parent_p_w(
                    word_n_id,
                    parent_p_w,
                    &mut self.whpylm.d_array,
                    &mut self.whpylm.theta_array,
                );
                self.whpylm_parent_p_w_cache[depth] = p_w;
                let child = (*cur_node).find_child_pyp(context, generate_if_not_found);
                if child.is_none() && return_middle_node == true {
                    return Some(cur_node);
                }
                parent_p_w = p_w;
                cur_node = child.unwrap();
            }
        }
        return Some(cur_node);
    }

    fn compute_g_0_of_word_at_index_n(
        &mut self,
        sentence_as_chars: &Vec<char>,
        word_begin_index: usize,
        word_end_index: usize,
        word_n_id: u64,
    ) -> f64 {
        if word_n_id == EOS {
            return self.chpylm.g_0;
        }

        let word_length = word_end_index - word_begin_index + 1;
        match self.whpylm_g_0_cache.entry(word_n_id) {
            Entry::Vacant(e) => {
                self.most_recent_word = produce_word_with_bow_and_eow(
                    sentence_as_chars,
                    word_begin_index,
                    word_end_index,
                );
                let word_length_with_symbols = word_length + 2;
                let p_w = self.chpylm.compute_p_w(&self.most_recent_word);
                if word_length > self.max_word_length {
                    // self.whpylm_g_0_cache[&word_n_id] = p_w;
                    e.insert(p_w);
                    return p_w;
                } else {
                    let p_k_given_chpylm = self.compute_p_k_given_chpylm(word_length);
                    let t = detect_word_type_substr(
                        sentence_as_chars,
                        word_begin_index,
                        word_end_index,
                    );
                    let lambda = self.lambda_for_types[t];
                    let poisson_sample = sample_poisson_k_lambda(word_length, lambda);
                    let g_0 = p_w / p_k_given_chpylm * poisson_sample;

                    // Very rarely the result will exceed 1.
                    if !(0.0 < g_0 && g_0 < 1.0) {
                        print!("The result exceeds 1!");
                        // for i in word_begin_index..word_end_index + 1 {
                        //     print!("{}", sentence_as_chars[i]);
                        // }
                    }

                    e.insert(g_0);
                    return g_0;
                }
            }
            Entry::Occupied(e) => e.get().clone(),
        }
    }

    fn compute_p_k_given_chpylm(&self, k: usize) -> f64 {
        if k > self.max_word_length {
            return 0.0;
        } else {
            return self.p_k_chpylm[k];
        }
    }
}

fn sample_poisson_k_lambda(k: usize, lambda: f64) -> f64 {
    let dist = Poisson::new(lambda).unwrap();
    dist.pmf(k as u64)
}