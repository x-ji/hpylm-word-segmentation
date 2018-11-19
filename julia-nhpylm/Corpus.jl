include("Def.jl")

#= Begin Vocabulary =#
"""
This struct keeps track of all the characters in the target corpus.

This is necessary because in the character CHPYLM, the G_0 needs to be calculated via a uniform distribution over all possible characters of the target language.
"""
mutable struct Vocabulary
    all_characters::Set{Char}
end

function add_character(vocab::Vocabulary, character::Char)
    push!(vocab.all_characters, character)
end

function get_num_characters(vocab::Vocabulary)
    return length(vocab.all_characters)
end

#= Begin Corpus =#
"""
This struct keeps track of all sentences from the corpus files, and optionally the "true" segmentations, if any.
"""
mutable struct Corpus
    sentence_list::Vector{String}
    segmented_word_list::Vector{Vector{String}}
end

function read_corpus(corpus::Corpus, istream::IOStream)
    # Should I strip all spaces?
    append!(corpus.sentence_list, [line for line in readlines(istream) if !isempty(line)])
end

function get_num_sentences(corpus::Corpus)
    return length(corpus.sentence_list)
end

function get_num_segmented_words(corpus::Corpus)
    return length(corpus.segmented_word_list)
end

# TODO: Still some functionalities related to the "true segmentation" stuff.

#= Begin Sentence =#
mutable struct Sentence
    num_segments::UInt
    "The length of the segments within this sentence."
    segments_lengths::Vector{UInt}
    segment_starting_positions::Vector{UInt}
    "I think this means whether this sentence contains supervised guidance/segmentation already or something like that."
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
    function sentence(sentence_string::String)
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
    function sentence(sentence_string::String, supervised:Bool)
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

# TODO: Write out the split methods
function split(segments_without_special_tokens::Vector{UInt}, num_segments_without_special_tokens::UInt)
    
end

# TODO: Apparently he wrote a custom hash function for the words? Might try to directly feed in strings instead of their hashes and see how the memory cost goes.
# Or just use the built-in hash method if we're to keep the original structure. I'm pretty sure that all the words in a language is not going to break the hashing process.
function get_substr_word_id(s::Sentence, start_index::UInt, end_index::UInt)
    return hash(s.sentence_string[start_index:end_index])
end

function get_substr(s::Sentence, start_index::UInt, end_index::UInt)
    return s.sentence_string[start_index:end_index]
end


#= End Sentence =#

#= Begin Dataset =#
mutable struct Dataset
    vocabulary::Vocabulary
    corpus::Corpus
    max_sentence_length::UInt
    avg_sentence_length::UInt
    num_segmented_words::UInt
    vocabulary::Vocabulary
    train_sentences::Vector{Sentence}
    dev_sentences::Vector{Sentence}
    function Dataset(corpus::Corpus, train_proportion::Float64)
        dataset = new()
        dataset.vocabulary = Vocabulary()
        dataset.corpus = corpus
        dataset.max_sentence_length = 0
        dataset.avg_sentence_length = 0
        corpus_length::UInt = 0

        sentence_indices = zeros(UInt, get_num_sentences(corpus))
        for i in 1:get_num_sentences(corpus)
            sentence_indices[i] = i
        end
        shuffle!(sentence_indices)

        # How much of the input data will be used for training vs. used as dev (is there even a dev set in tihs one?)
        train_proportion = min(1.0, max(0.0, train_proportion))
        num_train_sentences = get_num_sentences(corpus) * train_proportion
        for i in 1:get_num_sentences(corpus)
            sentence= corpus.sentence_list[sentence_indices[i]]
            if i <= num_train_sentences
                add_sentence(sentence, dataset.train_sentences)
            else
                add_sentence(sentence, dataset.dev_sentences)
            end
            
            if length(sentence) > dataset.max_sentence_length
                dataset.max_sentence_length = length(sentence)
            end

            corpus_length += length(sentence)
        end
        
        # TODO: Don't think we have any supervised data yet. Let's see this one later.
        # num_segmented_words = get_num_segmented_words(corpus)
        # for i in 1:num_segmented_words

        # end

        dataset.avg_sentence_length = corpus_length / get_num_sentences(corpus)

        return dataset
    end
end

function get_num_train_sentences(dataset::Dataset)
    return length(dataset.train_sentences)
end

function get_num_dev_sentences(dataset::Dataset)
    return length(dataset.dev_sentences)
end

function get_num_segmented_words(dataset::Dataset)
    return dataset.num_segmented_words
end

function add_sentence(dataset::Dataset, sentence_string::String, sentences::Vector{Sentence})
    @assert(length(sentence_string) > 0)
    for char in sentence_string
        add_character(dataset.vocabulary, char)
    end
    push!(sentences, Sentence(sentence_string))
end

# Maybe we can just get rid of these two getters which are there for no reason
# get_max_sentence_length and get_average_sentence_length



#= End Dataset =#