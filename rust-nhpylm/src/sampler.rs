use ndarray::{Array2, Array3, Array4};

use def::*;
use npylm::*;
use rand::prelude::*;
use rand::Rng;
use sentence::*;

pub struct Sampler {
    pub npylm: NPYLM,
    word_ids: Vec<u64>,
    substring_word_id_cache: Array2<u64>,
    alpha_tensor: Array3<f64>,
    p_w_h_cache: Array4<f64>,
    log_z: Vec<f64>,
    scaling_coefficients: Vec<f64>,
    backward_sampling_table: Vec<f64>,
    viterbi_backward_indices: Array3<usize>,
    max_word_length: usize,
    max_sentence_length: usize,
}

impl Sampler {
    pub fn new(npylm: NPYLM, max_word_length: usize, max_sentence_length: usize) -> Self {
        let size = max_sentence_length + 1;
        Self {
            npylm: npylm,
            word_ids: vec![0; 3],
            max_word_length: max_word_length,
            max_sentence_length: max_sentence_length,
            // TODO: Not sure if using 0.0 instead of something like undefined is the right choice. Let's see.
            log_z: vec![0.0; max_sentence_length + 1],
            scaling_coefficients: vec![0.0; size + 1],
            viterbi_backward_indices: Array3::zeros((
                max_sentence_length + 1,
                max_word_length + 1,
                max_word_length + 1,
            )),
            backward_sampling_table: vec![0.0; max_word_length * max_word_length],
            alpha_tensor: Array3::zeros((size + 1, max_word_length + 1, max_word_length + 1)),
            p_w_h_cache: Array4::zeros((
                max_sentence_length + 1,
                max_word_length + 1,
                max_word_length + 1,
                max_word_length + 1,
            )),
            substring_word_id_cache: Array2::zeros((max_sentence_length + 1, max_word_length + 1)),
        }
    }

    pub fn extend_capacity(&mut self, max_word_length: usize, max_sentence_length: usize) {
        if max_word_length <= self.max_word_length
            && max_sentence_length <= self.max_sentence_length
        {
            return;
        } else {
            self.allocate_capacity(max_word_length, max_sentence_length);
        }
    }

    fn allocate_capacity(&mut self, max_word_length: usize, max_sentence_length: usize) {
        let size = max_sentence_length + 1;
        self.max_word_length = max_word_length;
        self.max_sentence_length = max_sentence_length;
        // TODO = Not sure if using 0.0 instead of something like undefined is the right choice. Let's see.
        self.log_z = vec![0.0; max_sentence_length + 1];
        self.scaling_coefficients = vec![0.0; size + 1];
        self.viterbi_backward_indices = Array3::zeros((
            max_sentence_length + 1,
            max_word_length + 1,
            max_word_length + 1,
        ));
        self.backward_sampling_table = vec![0.0; max_word_length * max_word_length];
        self.alpha_tensor = Array3::zeros((size + 1, max_word_length + 1, max_word_length + 1));
        self.p_w_h_cache = Array4::zeros((
            max_sentence_length + 1,
            max_word_length + 1,
            max_word_length + 1,
            max_word_length + 1,
        ));
        self.substring_word_id_cache =
            Array2::zeros((max_sentence_length + 1, max_word_length + 1));
    }

    fn get_substring_word_id_at_t_k(&mut self, sentence: &Sentence, t: usize, k: usize) -> u64 {
        let mut word_id = self.substring_word_id_cache[[t, k]];
        if word_id == 0 {
            word_id = sentence.get_substr_word_id(t - k, k - 1);
            self.substring_word_id_cache[[t, k]] = word_id;
        }
        word_id
    }

    fn forward_filtering(&mut self, sentence: &Sentence, with_scaling: bool) {
        self.alpha_tensor[[0, 0, 0]] = 1.0;
        for t in 1..sentence.length() + 1 {
            let mut prod_scaling = 1.0;
            for k in 1..t.min(self.max_word_length) + 1 {
                if with_scaling && k > 1 {
                    prod_scaling *= self.scaling_coefficients[t - k + 1];
                }

                for j in if t == k { 0 } else { 1 }..(t - k).min(self.max_word_length) + 1 {
                    self.alpha_tensor[[t, k, j]] = 0.0;
                    self.calculate_alpha_t_k_j(sentence, t, k, j, prod_scaling);
                }
            }
        }
    }

    fn calculate_alpha_t_k_j(
        &mut self,
        sentence: &Sentence,
        t: usize,
        k: usize,
        j: usize,
        prod_scaling: f64,
    ) {
        let word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);
        // let sentence_as_chars = &sentence.characters;

        if j == 0 {
            self.word_ids[0] = BOS;
            self.word_ids[1] = BOS;
            self.word_ids[2] = word_k_id;
            let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                &sentence.characters,
                &self.word_ids,
                2,
                t - k,
                t - 1,
            );
            self.alpha_tensor[[t, k, 0]] = p_w_h * prod_scaling;
            self.p_w_h_cache[[t, k, 0, 0]] = p_w_h;
            return;
        } else if t - k - j == 0 {
            let word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
            self.word_ids[0] = BOS;
            self.word_ids[1] = word_j_id;
            self.word_ids[2] = word_k_id;
            let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                &sentence.characters,
                &self.word_ids,
                2,
                t - k,
                t - 1,
            );
            self.alpha_tensor[[t, k, j]] = p_w_h * self.alpha_tensor[[t - k, j, 0]] * prod_scaling;
            self.p_w_h_cache[[t, k, j, 0]] = p_w_h;
            return;
        } else {
            let mut sum = 0.0;
            for i in 1..self.max_word_length.min(t - k - j) + 1 {
                let word_i_id = self.get_substring_word_id_at_t_k(sentence, t - k - j, i);
                let word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
                self.word_ids[0] = word_i_id;
                self.word_ids[1] = word_j_id;
                self.word_ids[2] = word_k_id;

                let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                    &sentence.characters,
                    &self.word_ids,
                    2,
                    t - k,
                    t - 1,
                );
                self.p_w_h_cache[[t, k, j, i]] = p_w_h;
                sum += p_w_h * self.alpha_tensor[[t - k, j, i]];
            }

            self.alpha_tensor[[t, k, j]] = sum * prod_scaling;
            return;
        }
    }

    fn backward_sampling(&mut self, sentence: &Sentence) -> Vec<usize> {
        let mut t = sentence.length();
        let mut k = 0;
        let mut j = 0;
        let mut sum_length = 0;
        self.backward_sample_k_and_j(sentence, t, 1, &mut k, &mut j);

        let mut segment_lengths: Vec<usize> = Vec::new();
        segment_lengths.push(k);
        if j == 0 && k == t {
            return segment_lengths;
        }

        segment_lengths.push(j);
        t -= k + j;
        sum_length += k + j;
        let mut next_word_length = j;

        while t > 0 {
            if t == 1 {
                k = 1;
                j = 0;
            } else {
                self.backward_sample_k_and_j(sentence, t, next_word_length, &mut k, &mut j);
            }
            segment_lengths.push(k);
            t -= k;
            if j == 0 {
                assert!(t == 0);
            } else {
                assert!(j <= self.max_word_length);
                segment_lengths.push(j);
                t -= j;
            }
            sum_length += k + j;
            next_word_length = j;
        }
        segment_lengths.reverse();
        return segment_lengths;
    }

    fn backward_sample_k_and_j(
        &mut self,
        sentence: &Sentence,
        t: usize,
        third_gram_length: usize,
        sampled_k: &mut usize,
        sampled_j: &mut usize,
    ) {
        let mut table_index = 0;
        let sentence_length = sentence.length();
        let mut sum_p = 0.0;
        for k in 1..t.min(self.max_word_length) + 1 {
            for j in 1..(t - k).min(self.max_word_length) + 1 {
                let mut word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
                let mut word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);
                let mut word_t_id = EOS;
                if t < sentence_length {
                    word_t_id = self.get_substring_word_id_at_t_k(
                        sentence,
                        t + third_gram_length,
                        third_gram_length,
                    );
                }
                self.word_ids[0] = word_j_id;
                self.word_ids[1] = word_k_id;
                self.word_ids[2] = word_t_id;
                let mut p_w_h = 0.0;
                if t == sentence_length {
                    p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                        &sentence.characters,
                        &self.word_ids,
                        2,
                        t,
                        t,
                    );
                } else {
                    p_w_h = self.p_w_h_cache[[t + third_gram_length, third_gram_length, k, j]];
                }
                let p = p_w_h * self.alpha_tensor[[t, k, j]];
                self.backward_sampling_table[table_index] = p;
                sum_p += p;
                table_index += 1;
            }

            if t == k {
                let mut j = 0;
                let mut word_j_id = BOS;
                let mut word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);
                let mut word_t_id = EOS;
                if t < sentence_length {
                    word_t_id = self.get_substring_word_id_at_t_k(
                        sentence,
                        t + third_gram_length,
                        third_gram_length,
                    );
                }
                self.word_ids[0] = word_j_id;
                self.word_ids[1] = word_k_id;
                self.word_ids[2] = word_t_id;
                let mut p_w_h = 0.0;
                if t == sentence_length {
                    p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                        &sentence.characters,
                        &self.word_ids,
                        2,
                        t,
                        t,
                    );
                } else {
                    p_w_h = self.p_w_h_cache[[t + third_gram_length, third_gram_length, k, j]];
                }
                let p = p_w_h * self.alpha_tensor[[t, k, j]];
                self.backward_sampling_table[table_index] = p;
                sum_p += p;
                table_index += 1;
            }
        }

        let normalizer = 1.0 / sum_p;
        let randnum: f64 = rand::thread_rng().gen();
        let mut index = 0;
        let mut stack = 0.0;
        for k in 1..t.min(self.max_word_length) + 1 {
            for j in 1..(t - k).min(self.max_word_length) + 1 {
                stack += self.backward_sampling_table[index] * normalizer;
                if randnum < stack {
                    *sampled_k = k;
                    *sampled_j = j;
                    return;
                }
                index += 1;
            }

            if t == k {
                stack += self.backward_sampling_table[index] * normalizer;
                if randnum < stack {
                    *sampled_k = k;
                    *sampled_j = 0;
                    return;
                }
                index += 1;
            }
        }
        print!("Fell through?");
    }

    fn blocked_gibbs_segment(&mut self, sentence: &Sentence, with_scaling: bool) -> Vec<usize> {
        for i in 0..sentence.length() + 1 {
            for j in 0..self.max_word_length + 1 {
                self.substring_word_id_cache[[i, j]] = 0;
            }
        }

        self.forward_filtering(sentence, with_scaling);
        return self.backward_sampling(sentence);
    }

    fn viterbi_argmax_calculate_alpha_t_k_j(
        &mut self,
        sentence: &Sentence,
        t: usize,
        k: usize,
        j: usize,
    ) {
        let word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);

        if j == 0 {
            self.word_ids[0] = BOS;
            self.word_ids[1] = BOS;
            self.word_ids[2] = word_k_id;
            let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                &sentence.characters,
                &self.word_ids,
                2,
                t - k,
                t - 1,
            );
            // Here two are the differences compared with the non viterbi method.
            self.alpha_tensor[[t, k, 0]] = p_w_h.ln();
            self.viterbi_backward_indices[[t, k, 0]] = 0;
            return;
        } else if t - k - j == 0 {
            let word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
            self.word_ids[0] = BOS;
            self.word_ids[1] = word_j_id;
            self.word_ids[2] = word_k_id;
            let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                &sentence.characters,
                &self.word_ids,
                2,
                t - k,
                t - 1,
            );
            // Here two are the differences compared with the non viterbi method.
            self.alpha_tensor[[t, k, j]] = p_w_h.ln() + self.alpha_tensor[[t - k, j, 0]];
            self.viterbi_backward_indices[[t, k, j]] = 0;
            return;
        } else {
            // Here two are the differences compared with the non viterbi method.
            let mut max_log_p = 0.0;
            let mut argmax = 0;
            for i in 1..self.max_word_length.min(t - k - j) + 1 {
                let word_i_id = self.get_substring_word_id_at_t_k(sentence, t - k - j, i);
                let word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
                self.word_ids[0] = word_i_id;
                self.word_ids[1] = word_j_id;
                self.word_ids[2] = word_k_id;

                let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                    &sentence.characters,
                    &self.word_ids,
                    2,
                    t - k,
                    t - 1,
                );
                // Here are the differences compared with the non viterbi method.
                let temp = p_w_h.ln() + self.alpha_tensor[[t - k, j, i]];
                if (argmax == 0 || temp > max_log_p) {
                    argmax = i;
                    max_log_p = temp;
                }
            }

            self.alpha_tensor[[t, k, j]] = max_log_p;
            // We use the viterbi_backward_indices matrix to store the i value that maximizes the possibility of the trigram.
            self.viterbi_backward_indices[[t, k, j]] = argmax;
            return;
        }
    }

    fn viterbi_forward_filtering(&mut self, sentence: &Sentence) {
        for t in 1..sentence.length() + 1 {
            for k in 1..t.min(self.max_word_length) + 1 {
                // There is no j, i.e. the second gram is also BOS.
                if t == k {
                    self.viterbi_argmax_calculate_alpha_t_k_j(sentence, t, k, 0);
                }

                // Note that in the t==k case, we will have range 1:0 which is automatically empty, so the following code will not be run.
                for j in 1..(t - k).min(self.max_word_length) + 1 {
                    self.viterbi_argmax_calculate_alpha_t_k_j(sentence, t, k, j);
                }
            }
        }
    }

    fn viterbi_argmax_backward_sample_k_and_j_to_eos(
        &mut self,
        sentence: &Sentence,
        t: usize,
        third_gram_length: usize,
        argmax_k: &mut usize,
        argmax_j: &mut usize,
    ) {
        let mut max_log_p = 0.0;
        *argmax_k = 0;
        *argmax_j = 0;
        for k in 1..t.min(self.max_word_length) + 1 {
            for j in 1..(t - k).min(self.max_word_length) + 1 {
                let mut word_j_id = self.get_substring_word_id_at_t_k(sentence, t - k, j);
                let mut word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);
                self.word_ids[0] = word_j_id;
                self.word_ids[1] = word_k_id;
                self.word_ids[2] = EOS;
                let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                    &sentence.characters,
                    &self.word_ids,
                    2,
                    t,
                    t,
                );
                let temp = p_w_h.ln() + self.alpha_tensor[[t, k, j]];
                if *argmax_k == 0 || temp > max_log_p {
                    max_log_p = temp;
                    *argmax_k = k;
                    *argmax_j = j;
                }
            }

            if t == k {
                let mut word_j_id = BOS;
                let mut word_k_id = self.get_substring_word_id_at_t_k(sentence, t, k);
                let mut word_t_id = EOS;
                self.word_ids[0] = word_j_id;
                self.word_ids[1] = word_k_id;
                self.word_ids[2] = word_t_id;
                let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                    &sentence.characters,
                    &self.word_ids,
                    2,
                    t,
                    t,
                );
                let temp = p_w_h.ln() + self.alpha_tensor[[t, k, 0]];
                if *argmax_k == 0 || temp > max_log_p {
                    max_log_p = temp;
                    *argmax_k = k;
                    *argmax_j = 0;
                }
            }
        }
    }

    fn viterbi_backward_sampling(&mut self, sentence: &Sentence) -> Vec<usize> {
        let mut segment_lengths: Vec<usize> = Vec::new();
        let mut t = sentence.length();
        let mut sum_length = 0;
        let mut k = 0;
        let mut j = 0;
        self.viterbi_argmax_backward_sample_k_and_j_to_eos(sentence, t, 1, &mut k, &mut j);
        segment_lengths.push(k);
        sum_length += k;

        // There's only one word in total for the sentence.
        if j == 0 && k == t {
            return segment_lengths;
        }

        segment_lengths.push(j);
        let mut i = self.viterbi_backward_indices[[t, k, j]];

        sum_length += j + i;

        // Move the "sentence end" forward
        t -= k;
        k = j;
        j = i;

        if i == 0 {
            assert!(sum_length == sentence.length());
            segment_lengths.reverse();
            return segment_lengths;
        }

        segment_lengths.push(i);

        // TODO: This is more or less a repeat of the above. Should be able to refactor it?
        while t > 0 {
            i = self.viterbi_backward_indices[[t, k, j]];
            if i != 0 {
                segment_lengths.push(i);
            }
            t -= k;
            k = j;
            j = i;
            sum_length += i;
        }

        segment_lengths.reverse();
        return segment_lengths;
    }

    /// This function uses viterbi algorithm to sample the segmentation of a sentence, instead of the approach in the `blocked_gibbs_segment` function above. They should both be valid approaches.
    pub fn viterbi_decode(&mut self, sentence: &Sentence) -> Vec<usize> {
        self.alpha_tensor[[0, 0, 0]] = 0.0;
        self.log_z[0] = 0.0;
        for t in 0..sentence.length() + 1 {
            for k in 0..self.max_word_length + 1 {
                self.substring_word_id_cache[[t, k]] = 0;
            }
        }
        self.viterbi_forward_filtering(sentence);
        return self.viterbi_backward_sampling(sentence);
    }

    pub fn compute_log_forward_probability(
        &mut self,
        sentence: &Sentence,
        with_scaling: bool,
    ) -> f64 {
        self.enumerate_forward_variables(sentence, with_scaling);
        let t = sentence.length() + 1;
        if !with_scaling {
            let k = 1;
            let mut alpha_eos = 0.0;
            for j in 1..self.max_word_length.min(t - k) {
                alpha_eos += self.alpha_tensor[[t, k, j]];
            }
            return alpha_eos.ln();
        } else {
            let mut log_p_x = 0.0;
            for i in 1..t + 1 {
                log_p_x += (1.0 / self.scaling_coefficients[i]).ln();
            }
            return log_p_x;
        }
    }

    // TODO: This function is a duplicate of some of the functionalities that we already performed above. Should be able to put it somewhere.
    fn enumerate_forward_variables(&mut self, sentence: &Sentence, with_scaling: bool) {
        for i in 0..sentence.length() + 1 {
            for j in 0..self.max_word_length + 1 {
                self.substring_word_id_cache[[i, j]] = 0;
            }
        }

        self.forward_filtering(sentence, with_scaling);

        let mut alpha_eos = 0.0;
        let mut t = sentence.length() + 1;
        let mut k = 1;
        for j in 1..self.max_word_length.min(t - k) + 1 {
            let mut prob_sum = 0.0;
            for i in if t - k - j == 0 { 0 } else { 1 }..self.max_word_length.min(t - k - j) + 1 {
                self.word_ids[0] = self.get_substring_word_id_at_t_k(sentence, t - k - j, i);
                self.word_ids[1] = self.get_substring_word_id_at_t_k(sentence, t - k, j);
                self.word_ids[2] = EOS;
                let p_w_h = self.npylm.compute_p_w_of_nth_word_as_chars(
                    &sentence.characters,
                    &self.word_ids,
                    2,
                    t,
                    t,
                );
                prob_sum += p_w_h * self.alpha_tensor[[t - k, j, i]];
            }
            self.alpha_tensor[[t, k, j]] = prob_sum;
            alpha_eos += prob_sum;
        }
        if with_scaling {
            self.scaling_coefficients[t] = 1.0 / alpha_eos;
        }
    }
}
