use def::*;
use std::fmt;

// pub const BOS: u64 = calculate_hash(&BOS_CHAR);
// pub const EOS: u64 = calculate_hash(&EOS_CHAR);

#[derive(Clone)]
pub struct Sentence {
    pub num_segments: usize,
    pub segment_lengths: Vec<usize>,
    pub segment_begin_positions: Vec<usize>,
    pub supervised: bool,
    pub characters: Vec<char>,
    pub word_ids: Vec<u64>,
}

impl Sentence {
    pub fn new_from_string(sentence_string: String, supervised: bool) -> Self {
        return Self::new(sentence_string.chars().collect(), supervised);
    }

    pub fn new(characters: Vec<char>, supervised: bool) -> Self {
        let mut word_ids: Vec<u64> = vec![0; characters.len() + 3];
        word_ids[0] = BOS;
        word_ids[1] = BOS;
        word_ids[3] = EOS;

        let mut segment_lengths: Vec<usize> = vec![0; characters.len() + 3];
        segment_lengths[0] = 1;
        segment_lengths[1] = 1;
        segment_lengths[2] = characters.len();
        segment_lengths[3] = 1;

        let mut segment_begin_positions: Vec<usize> = vec![0; characters.len() + 3];
        segment_begin_positions[0] = 0;
        segment_begin_positions[1] = 0;
        segment_begin_positions[2] = 0;
        segment_begin_positions[3] = characters.len();

        Self {
            characters: characters,
            word_ids: word_ids,
            segment_lengths: segment_lengths,
            segment_begin_positions: segment_begin_positions,
            num_segments: 4,
            supervised: supervised,
        }
    }

    pub fn length(&self) -> usize {
        self.characters.len()
    }

    pub fn get_num_segments_without_special_tokens(&self) -> usize {
        self.num_segments - 3
    }

    pub fn get_nth_segment_length(&self, n: usize) -> usize {
        self.segment_lengths[n]
    }

    pub fn get_nth_word_id(&self, n: usize) -> u64 {
        self.word_ids[n]
    }

    pub fn get_substr_word_id(&self, start_index: usize, end_index: usize) -> u64 {
        // It seems that we need to + 1
        let substr: String = self.characters[start_index..end_index + 1]
            .into_iter()
            .collect();
        calculate_hash(&substr)
    }

    pub fn get_nth_word_chars(&self, n: usize) -> &[char] {
        // println!("Characters: {:?}", self.characters);
        if n < 2 {
            return &['<', 'B', 'O', 'S', '>'];
        } else {
            let start_position: usize = self.segment_begin_positions[n];
            let end_position: usize = start_position + self.segment_lengths[n];
            // Note that Rust's slicing is [a, b) interval, while Julia's is [a, b] interval!
            return &self.characters[start_position..end_position];
        }
    }

    pub fn split_sentence_with_num_segments(
        &mut self,
        segment_lengths: Vec<usize>,
        num_segments_without_special_tokens: usize,
    ) {
        let mut cur_start = 0;
        let mut sum_length = 0;
        let mut index = 0;

        while index < num_segments_without_special_tokens {
            sum_length += segment_lengths[index];
            let cur_length = segment_lengths[index];

            self.segment_lengths[index + 2] = cur_length;
            self.word_ids[index + 2] =
                self.get_substr_word_id(cur_start, cur_start + cur_length - 1);
            self.segment_begin_positions[index + 2] = cur_start;
            cur_start += cur_length;
            index += 1;
        }

        assert!(sum_length == self.characters.len());

        // EOS
        self.segment_lengths[index + 2] = 1;
        self.word_ids[index + 2] = EOS;
        self.segment_begin_positions[index + 2] = self.segment_begin_positions[index + 1];
        index += 1;

        while index < self.characters.len() {
            self.segment_lengths[index + 2] = 0;
            self.segment_begin_positions[index + 2] = 0;
            index += 1;
        }

        self.num_segments = num_segments_without_special_tokens + 3;
    }

    pub fn split_sentence(&mut self, segment_lengths: Vec<usize>) {
        let num_segments_without_special_tokens = segment_lengths.len();
        self.split_sentence_with_num_segments(segment_lengths, num_segments_without_special_tokens);
    }

    // pub fn print_sentence(&self, f: &mut fmt::Formatter) {
    //     for index in 2..self.num_segments - 1 {
    //         write!(f, "{}", self.get_nth_word_chars(index));
    //     }
    // }
}

impl fmt::Display for Sentence {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // let mut r: fmt::Result = std::result::Result::Err;
        for index in 2..self.num_segments - 1 {
            write!(f, "{:?}", self.get_nth_word_chars(index));
        }
        // ???
        write!(f, "")
    }
}
