import Base.length
import Base.show

# Let me try some string substitutes
const START = "ϵ"
# This is \Epsilon, not E! Not the same thing eh.
const STOP = "Ε"

# This exception was present but never used in the original vpyp code. But it has its use here.
struct OutOfVocabularyException <: Exception end

"""
A utility struct for the conversion between words and integer representations (ids).
"""
mutable struct Vocabulary
    "The mapping from original words to integers"
    word2id::Dict{String,Int}

    "The mapping from integers to original words"
    id2word::Array{String,1}

    "Whether new mappings can still be added to the vocabulary"
    frozen::Bool

    """
    Construct a `Vocabulary` struct. `start_stop` determines whether to initialize the mappings with special `START` and `STOP` symbols as `<s>` and `</s>`, which are normally used in corpora.
    """
    function Vocabulary(start_stop::Bool = true, init::Array{String,1} = Array{String,1}())
        v = new()
        # This doesn't seem to be useful anymore in the new program? Or not. Well we still do need to have a sentence stop token at least, right?
        # OK let me still keep them anyways. Doesn't really hurt either way.
        if start_stop
            v.word2id = Dict("<s>" => 0, "</s>" => 1)
            v.id2word = ["<s>", "</s>"]
        else
            v.word2id = Dict{String,Int}()
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
function get(v::Vocabulary, word::Int)::String
    return v.id2word[word]
end

"Get the integer representation for the word string."
function get(v::Vocabulary, word::String)::Int
    if !haskey(v.word2id, word)
        if v.frozen
            # This thing is actually not used since another way is used to keep track of word count.
            throw(OutOfVocabularyException())
        else
            # This order is correct since in Julia indices start from 1.
            # Actually push! is really memory intensive. If done only once during the initialization it's fine, otherwise it can be a problem.
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
    @assert(1 <= value <= length(v))
    v.id2word[value] = word
    v.word2id[word] = value
end

function set(v::Vocabulary, word::String, value::Int)
    @assert(1 <= value <= length(v))
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
function read_corpus(stream::IOStream, char_vocab::Vocabulary)
    # Now we need to construct the char_vocab, assuming that in the input everything is stuck together by default.
    # readlines returns a vector of strings
    # Discard empty lines
    # Make sure that everything is stuck together and then interpreted on a char-by-char basis.
    # The point is that in the target languages such as Chinese and Japanese, there should not be any spaces.
    # Currently we still need to convert the character to string.
    # And then we return the integer representation of the character via the vocabulary.
    return [[get(char_vocab, string(c)) for c in line if !Base.isspace(c)] for line in readlines(stream) if !isempty(line)]
end

# TODO: Maybe I can do some caching on this one. But it is indeed inherently made harder, because there could be new segmentations introduced each time.
# Yeah at least this is definitely cacheable when I'm producing character n-grams from word n-grams. Let's see.
# OK, changing everything from push! to preallocation really didn't save that much memory... Only an increase of about 10% it seems. The push! implementation seems to be reasonable enough? Then I should try to find other ways to improve the performance.
"Return all ngrams of the specified `order` from the given `sentence`. The return type is array of arrays of integers."
function ngrams(sentence::Array{String,1}, order::Int)
    # The deque from Python is really not the same as CircularDeque here. Doesn't support automatic replacement of elements whatsoever.
    # The original implementation in Python uses `yield` but I guess I won't need it here.

    sentence_size = length(sentence)
    temp::Array{String,1} = fill("",
        # The START symbols to be filled in at the beginning.
        order - 1
        # The actual sentence part
        + sentence_size
        # The STOP symbol to be filled in at the end.
        + 1)

    # temp = fill(START, order - 1)
    # Need the broadcasting operator to assign multiple indices at the same time.
    temp[1:order - 1] .= START

    # append!(temp, sentence)
    temp[order:order+sentence_size - 1] = sentence

    # push!(temp, STOP)
    temp[end] = STOP

    # output = Array{Array{String,1},1}()
    output::Array{Array{String,1},1} = fill(String[], length(temp) - order + 1)

    for index in 1:(length(temp) - order + 1)
        # ngram = Array{String, 1}()
        ngram::Array{String,1} = fill("", order)

        for i2 in index:index + (order - 1)
            # push!(ngram, temp[i2])
            # Julia's index starts from 1.
            ngram[i2-index + 1] = temp[i2]
        end
        # push!(output, ngram)
        output[index] = ngram
    end

    return output
end
