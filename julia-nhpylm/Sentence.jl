import Base.length

#= Begin Sentence =#
mutable struct Sentence
    num_segments::UInt
    "The length of the segments within this sentence."
    segments_lengths::Vector{UInt}
    segment_starting_positions::Vector{UInt}
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
    ERROR: StringIndexError("我们", 2)
    > a[4]
    '们': Unicode U+4eec (category Lo: Letter, other)

    Although of course we can directly iterate the string with `for c in a`. Maybe that will make for a more idiomatic solution in Julia. Let me see if I can refactor the code that way later then.
    ```

    Therefore, we can't do much with the sentence_string by trying to directly index-access it!
    """
    characters::Vector{Char}
    "The corresponding integer representations of the words. This includes both bos (2) and eos (1)"
    word_ids::Vector{UInt}
    "The string that makes up the sentence"
    sentence_string::String
    function Sentence(sentence_string::String)
        s = new()

        s.sentence_string = sentence_string
        s.characters = Vector{Char}(sentence_string)
        s.word_ids = zeros(UInt, length(sentence_string) + 3)
        s.segments_lengths = zeros(UInt, length(sentence_string) + 3)
        s.segment_starting_positions = zeros(UInt, length(sentence_string) + 3)

        # TODO: Optimize this process so that BOS and EOS tokens are already added when the sentences are read in.
        s.word_ids[1] = BOS
        s.word_ids[2] = BOS
        s.word_ids[3] = get_substr_word_id(1, length(sentence_string))
        s.word_ids[4] = EOS

        # Of course the lengths of BOS and EOS etc. are all 1.
        s.segment_lengths[1] = 1
        s.segment_lengths[2] = 1
        s.segment_lengths[3] = length(sentence_string)
        s.segment_lengths[4] = 1

        s.segment_starting_positions[1] = 1
        s.segment_starting_positions[2] = 1
        s.segment_starting_positions[3] = 1
        s.segment_starting_positions[4] = length(sentence_string)

        s.num_segments = 4

        s.supervised = false

        return s
    end
    function sentence(sentence_string::String, supervised::Bool)
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

function get_nth_segment_length(s::Sentence, n::UInt)
    @assert(n <= s.num_segments)
    return s.segment_lengths[n]
end

function get_nth_word_id(s::Sentence, n::UInt)
    @assert(n <= s.num_segments)
    return s.word_ids[n]
end

function get_nth_word_string(s::Sentence, n::UInt)
    # The last segment is <EOS>
    @assert(n < s.num_segments - 1)
    # TODO: This is all hard-coded. We'll need to change them if we're to support bigrams.
    # Can't we make the code a bit more generic? Don't think it would be that hard eh.
    if n <= 2
        return "<BOS>"
    else
        start_position = s.segment_starting_positions[n]
        end_position = start_position + s.segment_lengths[n]
        return s.sentence_string[start_position:end_position]
    end
end

# Hopefully this wouldn't be too slow.
function Base.show(io::IO, s::Sentence)
    for index in 3:s.num_segments - 1
        print(io, get_nth_word_string(s, index))
        print(" / ")
    end
end

# This method is to split the sentence using an already calculated segment_lengths vector, which contains the lengths of each segment.
# Note that the segment_lengths array is without containing any BOS or EOS tokens.
function split_sentence(sentence::Sentence, segment_lengths::Vector{UInt}, num_segments_without_special_tokens::UInt)
    cur_start = 1
    index = 1
    while index < num_segments_without_special_tokens
        cur_length = segment_lengths[index]
        sentence.segment_lengths[index + 2] = cur_length
        sentence.word_ids[index + 2] = get_substr_word_id(sentence, cur_start, cur_start + cur_length - 1)
        sentence.segment_starting_positions[index + 2] = cur_start
        cur_start += cur_length
        index += 1
    end
    # Also need to take care of EOS now that the actual string ended.
    sentence.segment_lengths[index + 2] = 1
    sentence.word_ids[index + 2] = EOS
    # So the EOS token is considered to be a part of the last word?
    sentence.segment_starting_positions[index + 2] = sentence.segment_starting_positions[index + 1]
    index += 1
    # Yeah we did initialize the lengths of those things to be length(sentence_string) + 3, but weren't they initialized to 0s from the very beginning?
    # for n in index:length(sentence.sentence_string)
    #     sentence.segment_lengths[n + 2] = 0
    #     sentence.segment_starting_positions[n + 2] = 0
    # end
    sentence.num_segments = num_segments_without_special_tokens + 3
end


function split_sentence(sentence::Sentence, segment_lengths::Vector{UInt})
    num_segments_without_special_tokens = length(segment_lengths)
    split_sentence(sentence, segment_lengths, num_segments_without_special_tokens)
end

# TODO: Apparently he wrote a custom hash function for the words? Might try to directly feed in strings instead of their hashes and see how the memory cost goes.
# Or just use the built-in hash method if we're to keep the original structure. I'm pretty sure that all the words in a language is not going to break the hashing process.
"Get the word id of the substring with start_index and end_index. Note that in Julia the end_index is inclusive."
function get_substr_word_id(s::Sentence, start_index::UInt, end_index::UInt)
    return hash(s.sentence_string[start_index:end_index])
end

function get_substr(s::Sentence, start_index::UInt, end_index::UInt)
    return s.sentence_string[start_index:end_index]
end
