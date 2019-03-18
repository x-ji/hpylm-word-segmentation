use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const BOS_CHAR: char = 'Α';
pub const EOS_CHAR: char = 'Ω';

// pub const BOS: u64 = calculate_hash(&BOS_CHAR);
// pub const EOS: u64 = calculate_hash(&EOS_CHAR);

pub struct Sentence {
  num_segments: usize,
  segment_lengths: Vec<usize>,
  segment_begin_positions: Vec<usize>,
  supervised: bool,
  characters: Vec<char>,
  word_ids: Vec<u64>,
  sentence_string: String,
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
  let mut s = DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

impl Sentence {
  pub fn new(sentence_string: String, supervised: bool) -> Self {
    let BOS = calculate_hash(&BOS_CHAR);
    let EOS = calculate_hash(&EOS_CHAR);

    let mut word_ids: Vec<u64> = vec![0; sentence_string.len() + 3];
    word_ids[0] = BOS;
    word_ids[1] = BOS;
    word_ids[3] = EOS;

    let mut segment_lengths: Vec<usize> = vec![0; sentence_string.len() + 3];
    segment_lengths[0] = 1;
    segment_lengths[1] = 1;
    segment_lengths[2] = sentence_string.len();
    segment_lengths[3] = 1;

    let mut segment_begin_positions: Vec<usize> = vec![0; sentence_string.len() + 3];
    segment_begin_positions[0] = 0;
    segment_begin_positions[1] = 0;
    segment_begin_positions[2] = 0;
    segment_begin_positions[3] = sentence_string.len();

    Self {
      characters: sentence_string.chars().collect(),
      sentence_string: sentence_string,
      word_ids: word_ids,
      segment_lengths: segment_lengths,
      segment_begin_positions: segment_begin_positions,
      num_segments: 4,
      supervised: supervised,
    }
  }
}

fn length(s: &Sentence) -> usize {
  s.sentence_string.len()
}

fn get_num_segments_without_special_tokens(s: &Sentence) -> usize {
  s.num_segments - 3
}

fn get_nth_segment_length(s: &Sentence, n: usize) -> usize {
  s.segment_lengths[n]
}

fn get_nth_word_id(s: &Sentence, n: usize) -> u64 {
  s.word_ids[n]
}

fn get_substr_word_id(s: &Sentence, start_index: usize, end_index: usize) -> u64 {
  // Why don't you just directly index the characters?
  // calculate_hash(&s.characters[start_index..end_index])
  let substr: String = s
    .sentence_string
    .chars()
    .take(end_index)
    .skip(start_index)
    .collect();
  calculate_hash(&substr)
}

fn get_nth_word_string(s: &Sentence, n: usize) -> String {
  if n < 2 {
    return String::from("<BOS>");
  } else {
    let start_position: usize = s.segment_begin_positions[n];
    let end_position: usize = start_position + s.segment_lengths[n];
    // Note that Rust's slicing is [a, b) interval, while Julia's is [a, b] interval!
    return s.characters[start_position..end_position]
      .iter()
      .cloned()
      .collect::<String>();
  }
}

fn split_sentence_with_num_segments(
  sentence: &mut Sentence,
  segment_lengths: Vec<usize>,
  num_segments_without_special_tokens: usize,
) {
  let EOS = calculate_hash(&EOS_CHAR);
  let mut cur_start = 0;
  let mut sum_length = 0;
  let mut index = 0;

  while index < num_segments_without_special_tokens {
    sum_length += segment_lengths[index];
    let cur_length = segment_lengths[index];

    sentence.segment_lengths[index + 2] = cur_length;
    sentence.word_ids[index + 2] =
      get_substr_word_id(sentence, cur_start, cur_start + cur_length - 1);
    sentence.segment_begin_positions[index + 2] = cur_start;
    cur_start += cur_length;
    index += 1;
  }

  // EOS
  sentence.segment_lengths[index + 2] = 1;
  sentence.word_ids[index + 2] = EOS;
  sentence.segment_begin_positions[index + 2] = sentence.segment_begin_positions[index + 1];
  index += 1;

  while index < sentence.sentence_string.len() {
    sentence.segment_lengths[index + 2] = 0;
    sentence.segment_begin_positions[index + 2] = 0;
    index += 1;
  }

  sentence.num_segments = num_segments_without_special_tokens + 3;
}

fn split_sentence(sentence: &mut Sentence, segment_lengths: Vec<usize>) {
  let num_segments_without_special_tokens = segment_lengths.len();
  split_sentence_with_num_segments(
    sentence,
    segment_lengths,
    num_segments_without_special_tokens,
  );
}
