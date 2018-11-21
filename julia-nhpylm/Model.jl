include("Def.jl")
include("NPYLM.jl")
include("Sampler.jl")
include("Corpus.jl")

"""
This is the struct that will server as a container for everything. it will be serialized after training.
"""

struct Model
    npylm::NPYLM
    sampler::Sampler
    function Model(dataset::Dataset, max_word_length::UInt)
        model = new()
        max_sentence_length = dataset.max_sentence_length
        # The G_0 probability for the character HPYLM, which depends on the number of different characters in the whole corpus.
        chpylm_G_0 = 1.0 / get_num_characters(dataset.vocabulary)
        model.npylm = NPYLM(max_word_length, max_sentence_length, chpylm_G_0, 4, 1, CHPYLM_β_STOP, CHPYLM_β_PASS)
        model.sampler = Sampler(model.npylm, max_word_length, max_sentence_length)
        return model
    end

    function Model(dataset::Dataset, max_word_length::UInt, initial_a, initial_b, chpylm_β_stop, chpylm_β_pass)
        model = new()
        max_sentence_length = dataset.max_sentence_length
        chpylm_G_0 = 1.0 / get_num_characters(dataset.vocabulary)
        model.npylm = NPYLM(max_word_length, max_sentence_length, chpylm_G_0, initial_a, initial_b, chpylm_β_stop, chpylm_β_pass)
        model.sampler = Sampler(model.npylm, max_word_length, max_sentence_length)
        return model
    end

    # Try to restore this thing from a file. Not sure if I'll need a particular method for that.
    # function Model(file::IOStream)
    #     model = new()
    #     model.npylm = NPYLM()
        
    #     return model
    # end
end

function get_max_word_length(model::Model)
    return model.npylm.max_word_length
end

function set_initial_a(model::Model, initial_a::Float64)
    model.npylm.λ_a = initial_a
    sample_λ_with_initial_params(model.npylm)
end

function set_initial_b(model::Model, initial_b::Float64)
    model.npylm.λ_b = initial_b
    sample_λ_with_initial_params(model.npylm)
end

function set_chpylm_beta_stop(model::Model, stop::Float64)
    model.npylm.chpylm.beta_stop = stop
end

function set_chpylm_beta_pass(model::Model, pass::Float64)
    model.npylm.chpylm.beta_pass = pass
end

function segment_sentence(model::Model, sentence_string::String)::Vector{String}
    extend_capacity(model.sampler, model.npylm.max_word_length, length(sentence_string))
    extend_capacity(model.npylm, length(sentence_string))
    segment_lengths = UInt[]
    segmented_sentence = String[]
    sentence = Sentence(sentence_string)
    # I don't really get the difference between the viterbi_ methods and the normal methods. Is it the case that the viterbi_ methods just do the segmentation directly without trying to further train the model? A bit weird indeed. Let's see further.
    viterbi_decode(model.sampler, sentence, segment_lengths)
    # This method is so fucking insane. Why not print out the sentences immediately anyways. What the fuck.
    # split(sentence, segment_lengths)
    
    # Skip the first two BOS in the sentence.
    # start_index = 3

    # No I mean, seriously, what the hell is the problem of directly trying to access the actual strings from the original sentence_string??? Fuck off.
    start_index = 1
    for (index, length) in enumerate(segment_lengths)
        word = sentence_string[start_index, start_index + length - 1]
        push!(segmented_sentence, word)
        start_index += length
    end

    return segmented_sentence
end

function compute_log_forward_probability(model::Model, sentence_string::String, with_scaling::Bool)
    extend_capacity(model.sampler, model.npylm.max_word_length, length(sentence_string))
    extend_capacity(model.npylm, length(sentence_string))
    sentence = Sentence(sentence_string)
    return compute_log_forward_probability(model.sampler, sentence, with_scaling)
end