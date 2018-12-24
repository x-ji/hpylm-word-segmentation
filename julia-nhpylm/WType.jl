using LegacyStrings
using OffsetArrays

include("CType.jl")

const WORDTYPE_NUM_TYPES = 9

const WORDTYPE_ALPHABET = 1
const WORDTYPE_NUMBER = 2
const WORDTYPE_SYMBOL = 3
const WORDTYPE_HIRAGANA = 4
const WORDTYPE_KATAKANA = 5
const WORDTYPE_KANJI = 6
const WORDTYPE_KANJI_HIRAGANA = 7
const WORDTYPE_KANJI_KATAKANA = 8
const WORDTYPE_OTHER = 9

# TODO: Some approaches seem a bit too manual here and maybe we can simplify the code a bit.
"This refers to the full-width dash used to mark long vowels in Hiragana/Katakana"
function is_dash(char::Char)::Bool
    if codepoint(char) == 0x30FC
        return true
    else
        return false
    end
end

function is_hiragana(char::Char)::Bool
    if detect_ctype(char) == CTYPE_HIRAGANA
        return true
    else
        # Could also be a dash to indicate a long vowel.
        return is_dash(char)
    end
end

function is_katakana(char::Char)::Bool
    t = detect_ctype(char)
    if t == CTYPE_KATAKANA
        return true
    elseif t == CTYPE_KATAKANA_PHONETIC_EXTENSIONS
        return true
    else
        # Could also be a dash to indicate a long vowel.
        return is_dash(char)
    end
end

function is_kanji(char::Char)::Bool
    t = detect_ctype(char)
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E)
        return true
    end
    if(t == CTYPE_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F)
        return true
    end
    if(t == CTYPE_CJK_RADICALS_SUPPLEMENT)
        return true
    end
    return false
end

function is_number(char::Char)::Bool
    cp = codepoint(char)
    t = detect_ctype(char)
    if(t == CTYPE_BASIC_LATIN)
        if(0x30 <= cp && cp <= 0x39)
            return true
        end
        return false
    end
    if(t == CTYPE_NUMBER_FORMS)
        return true
    end
    if(t == CTYPE_COMMON_INDIC_NUMBER_FORMS)
        return true
    end
    if(t == CTYPE_AEGEAN_NUMBERS)
        return true
    end
    if(t == CTYPE_ANCIENT_GREEK_NUMBERS)
        return true
    end
    if(t == CTYPE_COPTIC_EPACT_NUMBERS)
        return true
    end
    if(t == CTYPE_SINHALA_ARCHAIC_NUMBERS)
        return true
    end
    if(t == CTYPE_CUNEIFORM_NUMBERS_AND_PUNCTUATION)
        return true
    end
    return false
end

function is_alphabet(char::Char)::Bool
    cp = codepoint(char)
    if(0x41 <= cp && cp <= 0x5a)
        return true
    end
    if(0x61 <= cp && cp <= 0x7a)
        return true
    end
    return false
end

function is_symbol(char::Char)::Bool
    if(is_alphabet(char))
        return false
    end
    if(is_number(char))
        return false
    end
    if(is_kanji(char))
        return false
    end
    if(is_hiragana(char))
        return false
    end
    return true
end

"""
Detect the type of a word. Useful when we need to estimate different lambda values for different word types.

Basically it uses a heuristics to see the proportion of character types in this word to assign a word type eventually.
"""
function detect_word_type(word::UTF32String)::Int
    characters = Vector{Char}(word)
    offset_characters = OffsetArray(characters, 0:length(characters) - 1)
    return detect_word_type(offset_characters, 0, length(word) - 1)
end

function detect_word_type(sentence::OffsetVector{Char}, begin_index::Int, end_index::Int)::Int
    num_alphabet = 0
    num_number = 0
    num_symbol = 0
    num_hiragana = 0
    num_katakana = 0
    num_kanji = 0
    num_dash = 0

    word_length = end_index - begin_index + 1

    for i in begin_index:end_index
        cur_char = sentence[i]
        if(is_alphabet(cur_char))
            num_alphabet += 1
            continue
        end
        if(is_number(cur_char))
            num_number += 1
            continue
        end
        if(is_dash(cur_char))
            num_dash += 1
            continue
        end
        if(is_hiragana(cur_char))
            num_hiragana += 1
            continue
        end
        if(is_katakana(cur_char))
            num_katakana += 1
            continue
        end
        if(is_kanji(cur_char))
            num_kanji += 1
            continue
        end
        num_symbol += 1
    end
    if num_alphabet == word_length
        return WORDTYPE_ALPHABET
    end
    if(num_number == word_length)
        return WORDTYPE_NUMBER
    end
    if(num_hiragana + num_dash == word_length)
        return WORDTYPE_HIRAGANA
    end
    if(num_katakana + num_dash == word_length)
        return WORDTYPE_KATAKANA
    end
    if(num_kanji == word_length)
        return WORDTYPE_KANJI
    end
    if(num_symbol == word_length)
        return WORDTYPE_SYMBOL
    end
    if(num_kanji > 0)
        if(num_hiragana + num_kanji == word_length)
            return WORDTYPE_KANJI_HIRAGANA
        end
        if(num_hiragana > 0)
            if(num_hiragana + num_kanji + num_dash == word_length)
                return WORDTYPE_KANJI_HIRAGANA
            end
        end
        if(num_katakana + num_kanji == word_length)
            return WORDTYPE_KANJI_KATAKANA
        end
        if(num_katakana)
            if(num_katakana + num_kanji + num_dash == word_length)
                return WORDTYPE_KANJI_KATAKANA
            end
        end
    end
    return WORDTYPE_OTHER
end