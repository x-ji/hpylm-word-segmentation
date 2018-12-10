import Base.length
using LegacyStrings
using OffsetArrays

#= Begin Sentence =#
mutable struct Sentence
    num_segments::Int
    "The length of the segments within this sentence."
    segment_lengths::OffsetVector{Int}
    segment_begin_positions::OffsetVector{Int}
    "Indicates whether the sentence contains the true segmentation already."
    supervised::Bool
    """
    The individual characters that make up this sentence

    You may well still need them in Julia, since in Julia the string indices are byte indices, not real character indices. For example:

    ```
    > a = "我们"
    > a[1]
    '我': Unicode U+6211 (category Lo: Letter, other)
    > a[2]
    ERROR: UTF32StringIndexError("我们", 2)
    > a[4]
    '们': Unicode U+4eec (category Lo: Letter, other)

    Although of course we can directly iterate the string with `for c in a`. Maybe that will make for a more idiomatic solution in Julia. Let me see if I can refactor the code that way later then.
    ```

    Therefore, we can't do much with the sentence_string by trying to directly index-access it!
    """
    characters::OffsetVector{Char}
    "The corresponding integer representations of the words. This includes both bos (2) and eos (1)
    
    Because `hash` returns UInt, the contents also need to be UInt.
    "
    word_ids::OffsetVector{UInt}
    "The string that makes up the sentence"
    sentence_string::UTF32String
    function Sentence(sentence_string::UTF32String)
        s = new()

        s.sentence_string = sentence_string
        characters = Vector{Char}(sentence_string)
        s.characters = OffsetArray(characters, 0:length(characters) - 1)
        s.word_ids = zeros(UInt, 0:length(sentence_string) + 2)
        s.segment_lengths = zeros(Int, 0:length(sentence_string) + 2)
        s.segment_begin_positions = zeros(Int, 0:length(sentence_string) + 2)

        # TODO: Optimize this process so that BOS and EOS tokens are already added when the sentences are read in.
        s.word_ids[0] = BOS
        s.word_ids[1] = BOS
        # println("sentence_string: $(sentence_string), length of sentence_string: $(length(sentence_string))")
        s.word_ids[2] = get_substr_word_id(s, 0, length(sentence_string) - 1)
        s.word_ids[3] = EOS

        # Of course the lengths of BOS and EOS etc. are all 1.
        s.segment_lengths[0] = 1
        s.segment_lengths[1] = 1
        s.segment_lengths[2] = length(sentence_string)
        s.segment_lengths[3] = 1

        s.segment_begin_positions[0] = 0
        s.segment_begin_positions[1] = 0
        s.segment_begin_positions[2] = 0
        # Yeah if I wanted the 1-based indexing to work then this should have been length + 1 right? So indeed tons of bugs if the indexing systems are mixed up. Fuck that. Let me just try to stick to the original indexing system as much as possible then.
        s.segment_begin_positions[3] = length(sentence_string)

        s.num_segments = 4

        s.supervised = false

        return s
    end

    function sentence(sentence_string::UTF32String, supervised::Bool)
        sentence = Sentence(sentence_string)
        sentence.supervised = supervised
        return sentence
    end
end

function length(s::Sentence)
    return length(s.sentence_string)
end

# TODO: Will need refactoring if this is intended for two-grams
function get_num_segments_without_special_tokens(s::Sentence)
    return s.num_segments - 3
end

function get_nth_segment_length(s::Sentence, n::Int)
    @assert(n <= s.num_segments)
    return s.segment_lengths[n]
end

function get_nth_word_id(s::Sentence, n::Int)
    @assert(n <= s.num_segments)
    return s.word_ids[n]
end

# Or just use the built-in hash method if we're to keep the original structure. I'm pretty sure that all the words in a language is not going to break the hashing process.
"""
Get the word id of the substring with start_index and end_index. Note that in Julia the end_index is inclusive.

Note that the `hash` method returns UInt! This makes sense because a 2-fold increase in potential hash values can actually help a lot.
"""
function get_substr_word_id(s::Sentence, start_index::Int, end_index::Int)::UInt
    # println("Start index: $(start_index), end index: $(end_index), length of the sentence: $(length(s))")
    # Let me put +1 on everything involving sentence_string since I can't seem to change its indexing easily
    return hash(s.sentence_string[start_index+1:end_index+1])
end

function get_substr_word_string(s::Sentence, start_index::Int, end_index::Int)
    # Let me put +1 on everything involving sentence_string since I can't seem to change its indexing easily
    return s.sentence_string[start_index+1:end_index+1]
end

function get_nth_word_string(s::Sentence, n::Int)
    # The last segment is <EOS>
    @assert(n < s.num_segments)
    # TODO: This is all hard-coded. We'll need to change them if we're to support bigrams.
    # Can't we make the code a bit more generic? Don't think it would be that hard eh.
    if n < 2
        return "<BOS>"
    else
        @assert n < s.num_segments - 1
        start_position = s.segment_begin_positions[n]
        end_position = start_position + s.segment_lengths[n]
        return s.sentence_string[start_position+1:end_position+1]
    end
end

# Hopefully this wouldn't be too slow.
function Base.show(io::IO, s::Sentence)
    for index in 2:s.num_segments - 2
        print(io, get_nth_word_string(s, index))
        print(" / ")
    end
end

# This method is to split the sentence using an already calculated segment_lengths vector, which contains the lengths of each segment.
# Note that the segment_lengths array is without containing any BOS or EOS tokens.
function split_sentence(sentence::Sentence, segment_lengths::OffsetVector{Int})
    num_segments_without_special_tokens = length(segment_lengths)
    cur_start = 0
    sum_length = 0
    index = 0
    while index < num_segments_without_special_tokens - 1
        @assert segment_lengths[index] > 0
        sum_length += segment_lengths[index]

        cur_length = segment_lengths[index]

        # + 2 because the first two tokens are BOS.
        sentence.segment_lengths[index + 2] = cur_length
        sentence.word_ids[index + 2] = get_substr_word_id(sentence, cur_start, cur_start + cur_length - 1)
        sentence.segment_begin_positions[index + 2] = cur_start
        cur_start += cur_length
        index += 1
    end
    @assert sum_length == length(sentence.sentence.string)
    # Also need to take care of EOS now that the actual string ended.
    sentence.segment_lengths[index + 2] = 1
    sentence.word_ids[index + 2] = EOS
    # So the EOS token is considered to be a part of the last word?
    sentence.segment_begin_positions[index + 2] = sentence.segment_begin_positions[index + 1]
    index += 1
    # If those values were set previously, set them again to 0
    while index < length(sentence.sentence_string)
        sentence.segment_lengths[index + 2] = 0
        sentence.segment_begin_positions[index + 2] = 0
    end
    sentence.num_segments = num_segments_without_special_tokens + 3
end