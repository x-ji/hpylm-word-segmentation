# module Corpus
import Base.length
import Base.show

# Corpus
const START = 1
const STOP = 2

# This exception was present but never used in the original vpyp code.
struct OutOfVocabularyException <: Exception end

"""
A utility struct for the conversion between words and integer representations (ids).
"""
mutable struct Vocabulary
    "The mapping from original words to integers"
    word2id::Dict{String, Int}

    "The mapping from integers to original words"
    id2word::Array{String, 1}

    "Whether new mappings can still be added to the vocabulary"
    frozen::Bool

    """
    Construct a `Vocabulary` struct. `start_stop` determines whether to initialize the mappings with special `START` and `STOP` symbols as `<s>` and `</s>`, which are normally used in corpora.
    """
    function Vocabulary(start_stop::Bool=true, init::Array{String, 1}=Array{String, 1}())
        v = new()
        if start_stop
            v.word2id = Dict("<s>" => START, "</s>" => STOP)
            v.id2word = ["<s>", "</s>"]
        else
            v.word2id = Dict{String, Int}()
            v.id2word = []
        end
        v.frozen = false

        if !isempty(init)
            for word in init
                get(v, word)
            end
        end
        return v
    end
end

"Return the size of the vocabulary"
function length(v::Vocabulary)
    return length(v.id2word)
end

"Get the word string corresponding to the integer representation."
function get(v::Vocabulary, word::Int)
    return v.id2word[word]
end

"Get the integer representation for the word string."
function get(v::Vocabulary, word::String)
    if !haskey(v.word2id, word)
        if v.frozen
            # This thing is actually not used since another way is used to keep track of word count.
            throw(OutOfVocabularyException())
        else
            # This order is correct since in Julia indices start from 1.
            push!(v.id2word, word)
            v.word2id[word] = length(v)
        end
    end

    return v.word2id[word]
end

# So there are two versions of this function
# The naming of the variables should also be the reverse right?
"Associate the given integer with the given string."
function set(v::Vocabulary, value::Int, word::String)
    assert(1 <= value <= length(v))
    v.id2word[value] = word
    v.word2id[word] = value
end

function set(v::Vocabulary, word::String, value::Int)
    assert(1 <= value <= length(v))
    v.id2word[value] = word
    v.word2id[word] = value
end

# I'm actually not sure how useful this is though. Why don't they just directly change the reference anyways...
"Replace `Vocabulary` `v` with another `Vocabulary` `v2`"
function update(v::Vocabulary, v2::Vocabulary)
    v.word2id = v2.word2id
    v.id2word = v2.id2word
end

"Construct a `Vocabulary` struct from an input stream."
function read_corpus(stream::IOStream, vocabulary::Vocabulary)
    # This seems a weird way to do it. Though not sure if we can do it better.
    return [[get(vocabulary, String(word)) for word in split(line)] for line in readlines(stream)]
end

"Return all ngrams of the specified `order` from the given `sentence`. The return type is array of arrays of integers."
function ngrams(sentence::Array{Int, 1}, order::Int)
    # The deque from Python is really not the same as CircularDeque here. Doesn't support automatic replacement of elements whatsoever.
    # The original implementation in Python uses `yield` but I guess I won't need it here.

    output = Array{Array{Int, 1}, 1}()

    temp = fill(START, order-1)
    append!(temp, sentence)
    append!(temp, STOP)

    for index in 1:(length(temp) - order + 1)
        ngram = Array{Int, 1}()
        for i2 in index:index + (order - 1)
            push!(ngram, temp[i2])
        end
        # We're directly pushing an array here. Should achieve the same effect.
        push!(output, ngram)
        # push!(output, tuple(ngram...))
    end

    return output
end
# end
