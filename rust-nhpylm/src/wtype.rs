use ctype::*;

const WORDTYPE_NUM_TYPES: usize = 9;

const WORDTYPE_ALPHABET: usize = 1;
const WORDTYPE_NUMBER: usize = 2;
const WORDTYPE_SYMBOL: usize = 3;
const WORDTYPE_HIRAGANA: usize = 4;
const WORDTYPE_KATAKANA: usize = 5;
const WORDTYPE_KANJI: usize = 6;
const WORDTYPE_KANJI_HIRAGANA: usize = 7;
const WORDTYPE_KANJI_KATAKANA: usize = 8;
const WORDTYPE_OTHER: usize = 9;

pub fn is_dash(c: char) -> bool {
  if c as u32 == 0x30FC {
    return true;
  }
  return false;
}

pub fn is_hiragana(c: char) -> bool {
  let t = detect_ctype(c);
  if t == CTYPE_HIRAGANA {
    return true;
  }
  // Could also be a dash to indicate a long vowel.
  return is_dash(c);
}

pub fn is_katakana(c: char) -> bool {
  let t = detect_ctype(c);
  if t == CTYPE_KATAKANA {
    return true;
  }
  if t == CTYPE_KATAKANA_PHONETIC_EXTENSIONS {
    return true;
  }
  // Could also be a dash to indicate a long vowel.
  return is_dash(c);
}

pub fn is_kanji(c: char) -> bool {
  let t = detect_ctype(c);
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E) {
    return true;
  }
  if (t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F) {
    return true;
  }
  if (t == CTYPE_CJK_RADICALS_SUPPLEMENT) {
    return true;
  }
  return false;
}

pub fn is_number(c: char) -> bool {
  let t = detect_ctype(c);
  let character = c as u32;
  if (t == CTYPE_BASIC_LATIN) {
    if (0x30 <= character && character <= 0x39) {
      return true;
    }
    return false;
  }
  if (t == CTYPE_NUMBER_FORMS) {
    return true;
  }
  if (t == CTYPE_COMMON_INDIC_NUMBER_FORMS) {
    return true;
  }
  if (t == CTYPE_AEGEAN_NUMBERS) {
    return true;
  }
  if (t == CTYPE_ANCIENT_GREEK_NUMBERS) {
    return true;
  }
  if (t == CTYPE_COPTIC_EPACT_NUMBERS) {
    return true;
  }
  if (t == CTYPE_SINHALA_ARCHAIC_NUMBERS) {
    return true;
  }
  if (t == CTYPE_CUNEIFORM_NUMBERS_AND_PUNCTUATION) {
    return true;
  }
  return false;
}

pub fn is_alphabet(c: char) -> bool {
  let character = c as u32;
  if (0x41 <= character && character <= 0x5a) {
    return true;
  }
  if (0x61 <= character && character <= 0x7a) {
    return true;
  }
  return false;
}

pub fn is_symbol(c: char) -> bool {
  if (is_alphabet(c)) {
    return false;
  }
  if (is_number(c)) {
    return false;
  }
  if (is_kanji(c)) {
    return false;
  }
  if (is_hiragana(c)) {
    return false;
  }
  return true;
}

pub fn detect_word_type(word: &String) -> usize {
  return detect_word_type_substr(word.chars().collect(), 0, word.len() - 1);
}

pub fn detect_word_type_substr(chars: Vec<char>, start: usize, end: usize) -> usize {
  let mut num_alphabet = 0;
  let mut num_number = 0;
  let mut num_symbol = 0;
  let mut num_hiragana = 0;
  let mut num_katakana = 0;
  let mut num_kanji = 0;
  let mut num_dash = 0;
  let size = end - start + 1;
  for i in start..end + 1 {
    let target = chars[i];
    if is_alphabet(target) {
      num_alphabet += 1;
      continue;
    }
    if (is_number(target)) {
      num_number += 1;
      continue;
    }
    if (is_dash(target)) {
      num_dash += 1;
      continue;
    }
    if (is_hiragana(target)) {
      num_hiragana += 1;
      continue;
    }
    if (is_katakana(target)) {
      num_katakana += 1;
      continue;
    }
    if (is_kanji(target)) {
      num_kanji += 1;
      continue;
    }
    num_symbol += 1;
  }
  if (num_alphabet == size) {
    return WORDTYPE_ALPHABET;
  }
  if (num_number == size) {
    return WORDTYPE_NUMBER;
  }
  if (num_hiragana + num_dash == size) {
    return WORDTYPE_HIRAGANA;
  }
  if (num_katakana + num_dash == size) {
    return WORDTYPE_KATAKANA;
  }
  if (num_kanji == size) {
    return WORDTYPE_KANJI;
  }
  if (num_symbol == size) {
    return WORDTYPE_SYMBOL;
  }
  if (num_kanji > 0) {
    if (num_hiragana + num_kanji == size) {
      return WORDTYPE_KANJI_HIRAGANA;
    }
    if (num_hiragana > 0) {
      if (num_hiragana + num_kanji + num_dash == size) {
        return WORDTYPE_KANJI_HIRAGANA;
      }
    }
    if (num_katakana + num_kanji == size) {
      return WORDTYPE_KANJI_KATAKANA;
    }
    if (num_katakana > 0) {
      if (num_katakana + num_kanji + num_dash == size) {
        return WORDTYPE_KANJI_KATAKANA;
      }
    }
  }
  return WORDTYPE_OTHER;
}
