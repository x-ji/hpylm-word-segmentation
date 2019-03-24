use corpus::*;
use def::*;
use npylm::NPYLM;
use sampler::*;
use sentence::*;

pub struct Model {
    // I'm not sure if this struct is actually meaningful... Anyways let's refactor it later.
    pub sampler: Sampler,
}

impl Model {
    pub fn new(dataset: &Dataset, max_word_length: usize) -> Self {
        let max_sentence_length = dataset.max_sentence_length;
        let chpylm_g_0 = 1.0 / dataset.vocabulary.get_num_characters() as f64;
        let npylm = NPYLM::new(
            max_word_length,
            max_sentence_length,
            chpylm_g_0,
            4.0,
            1.0,
            CHPYLM_BETA_STOP,
            CHPYLM_BETA_PASS,
        );
        let sampler = Sampler::new(npylm, max_word_length, max_sentence_length);
        Self { sampler: sampler }
    }

    pub fn new_with_explicit_params(
        dataset: &Dataset,
        max_word_length: usize,
        initial_a: f64,
        initial_b: f64,
        chpylm_beta_stop: f64,
        chpylm_beta_pass: f64,
    ) -> Self {
        let max_sentence_length = dataset.max_sentence_length;
        let chpylm_g_0 = 1.0 / dataset.vocabulary.get_num_characters() as f64;
        let npylm = NPYLM::new(
            max_word_length,
            max_sentence_length,
            chpylm_g_0,
            initial_a,
            initial_b,
            chpylm_beta_stop,
            chpylm_beta_pass,
        );
        let sampler = Sampler::new(npylm, max_word_length, max_sentence_length);
        Self { sampler: sampler }
    }

    pub fn get_max_word_length(&self) -> usize {
        return self.sampler.npylm.max_word_length;
    }

    pub fn set_initial_a(&mut self, initial_a: f64) {
        self.sampler.npylm.lambda_a = initial_a;
        self.sampler.npylm.sample_lambda_with_initial_params();
    }

    pub fn set_initial_b(&mut self, initial_b: f64) {
        self.sampler.npylm.lambda_b = initial_b;
        self.sampler.npylm.sample_lambda_with_initial_params();
    }

    pub fn set_chpylm_beta_stop(&mut self, stop: f64) {
        self.sampler.npylm.chpylm.beta_stop = stop;
    }

    pub fn set_chpylm_beta_pass(&mut self, pass: f64) {
        self.sampler.npylm.chpylm.beta_pass = pass;
    }

    pub fn segment_sentence(&mut self, sentence_string: String) -> Vec<String> {
        // This is a bit silly...
        let max_word_length = self.sampler.npylm.max_word_length.clone();
        self.sampler
            .extend_capacity(max_word_length, sentence_string.len());

        self.sampler.npylm.extend_capacity(sentence_string.len());

        let mut segmented_sentence: Vec<String> = Vec::new();

        let mut sentence = Sentence::new(sentence_string, false);
        let segment_lengths = self.sampler.viterbi_decode(&sentence);

        // This is really convoluted. Let's see if we can do better.
        sentence.split_sentence(segment_lengths);
        for i in 0..sentence.get_num_segments_without_special_tokens() {
            segmented_sentence.push(sentence.get_nth_word_string(i + 2));
        }

        segmented_sentence
    }

    pub fn compute_log_forward_probability(
        &mut self,
        sentence_string: String,
        with_scaling: bool,
    ) -> f64 {
        let max_word_length = self.sampler.npylm.max_word_length.clone();
        self.sampler
            .extend_capacity(max_word_length, sentence_string.len());

        self.sampler.npylm.extend_capacity(sentence_string.len());

        let sentence = Sentence::new(sentence_string, false);

        return self
            .sampler
            .compute_log_forward_probability(&sentence, with_scaling);
    }
}
