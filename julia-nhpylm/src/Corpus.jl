# include("Def.jl")
# include("Sentence.jl")

#= Begin Vocabulary =#
"""
This struct keeps track of all the characters in the target corpus.

This is necessary because in the character CHPYLM, the G_0 needs to be calculated via a uniform distribution over all possible characters of the target language.
"""
struct Vocabulary
    all_characters::Set{Char}
    function Vocabulary()
        # Use `new` to access the normal Vocabulary constructor
        return new(Set{Char}())
    end
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
struct Corpus
    sentence_list::Vector{UTF32String}
    segmented_word_list::Vector{Vector{UTF32String}}
    function Corpus()
        # Use `new` to access the constructor
        return new(Vector{UTF32String}(), Vector{Vector{UTF32String}}())
    end
end

"""
Add an individual sentence to the corpus
"""
function add_sentence(corpus::Corpus, sentence_string::UTF32String)
    push!(corpus.sentence_list, sentence_string)
end

"""
Read the corpus from an input stream
"""
function read_corpus(corpus::Corpus, istream::IOStream)
    # Should I strip all spaces?
    append!(corpus.sentence_list, [line for line in readlines(istream) if !isempty(line)])
end

function get_num_sentences(corpus::Corpus)
    return length(corpus.sentence_list)
end

function get_num_already_segmented_sentences(corpus::Corpus)
    return length(corpus.segmented_word_list)
end

# TODO: Some functions related to the trainer which should perhaps be clarified later.

#= Begin Dataset =#
"""
This struct holds all the structs related to a session/task, including the vocabulary, the corpus and the sentences produced from the corpus.
"""
mutable struct Dataset
    vocabulary::Vocabulary
    corpus::Corpus
    "Max allowed sentence length in this dataset"
    max_sentence_length::Int
    "Average sentence length in this dataset"
    avg_sentence_length::Float64
    num_segmented_words::Int
    train_sentences::Vector{Sentence}
    dev_sentences::Vector{Sentence}
    function Dataset(corpus::Corpus, train_proportion::Float64)
        dataset = new()
        dataset.vocabulary = Vocabulary()
        dataset.corpus = corpus
        dataset.max_sentence_length = 0
        dataset.avg_sentence_length = 0
        dataset.train_sentences = Vector{Sentence}()
        dataset.dev_sentences = Vector{Sentence}()
        corpus_length::Int = 0

        sentence_indices = zeros(Int, get_num_sentences(corpus))
        for i in 1:get_num_sentences(corpus)
            sentence_indices[i] = i
        end
        shuffle!(sentence_indices)

        # How much of the input data will be used for training vs. used as dev (is there even a dev set in tihs one?)
        train_proportion = min(1.0, max(0.0, train_proportion))
        num_train_sentences = get_num_sentences(corpus) * train_proportion
        for i in 1:get_num_sentences(corpus)
            sentence_string = corpus.sentence_list[sentence_indices[i]]
            if i <= num_train_sentences
                add_sentence(dataset, sentence_string, dataset.train_sentences)
            else
                add_sentence(dataset, sentence_string, dataset.dev_sentences)
            end
            
            if length(sentence_string) > dataset.max_sentence_length
                dataset.max_sentence_length = length(sentence_string)
            end

            corpus_length += length(sentence_string)
        end
        
        # Is this actually sentences or segments?
        num_supervised_sentences = get_num_already_segmented_sentences(corpus)

        for i in 1:num_supervised_sentences
            # Reproduce the already segmented sentences
            words = corpus.word_sequence_list[i]
            segment_lengths = Vector{Int}()
            sentence_string = ""

            for word in words
                sentence_string += word
                push!(segment_lengths, length(word))
            end

            for char in sentence_string
                add_character(dataset.vocabulary, char)
            end

            sentence = Sentence(sentence_string, true)
            split_sentence(sentence, segment_lengths)
            push!(dataset.train_sentences, sentence)
            
            if (length(sentence_string) > dataset.max_sentence_length)
                dataset.max_sentence_length = length(sentence_string)
            end
            
            corpus_length += length(sentence_string)
        end

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

"Add a sentence to the train or dev sentence vector of the dataset"
function add_sentence(dataset::Dataset, sentence_string::UTF32String, sentences::Vector{Sentence})
    @assert(length(sentence_string) > 0)
    for char in sentence_string
        add_character(dataset.vocabulary, char)
    end
    push!(sentences, Sentence(sentence_string))
end
#= End Dataset =#