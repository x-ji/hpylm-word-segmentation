include("NPYLM.jl")
"""
This structs holds all the necessary fields and functions for sampling sentence segmentations using forward-backward inference.
"""
mutable struct Sampler
    npylm::NPYLM
    word_ids::Vector{UInt}
    "Cache of ids of some words previously segmented in this sentence."
    substring_word_id_cache::Array{UInt, 2}
    "3-dimensional tensor that contains the forward variables, i.e. in α[t][k][j] at p.104 of the paper"
    α_tensor::Array{Float64, 3}
    """
    Cache of word probabilities p_w given by the CHPYLM. Caching the value is useful so that we can avoid excessive repeated computations.
    
    I think the "h" here actually stands for Theta or something
    """
    p_w_h::Array{Float64, 4}
    "Normalization constants"
    log_z::Vector{Float64}
    scaling_coefficients::Vector{Float64}
    """
    Table to temporarily hold possibilities for the sampling of j and k during the backward sampling.

    It's called a "table" because we first record the probabilities for all candidates j and k values. After that, we need to *draw* from this "table" actual j and k values.

    See line 8 of Figure 5 of the paper.
    """
    backward_sampling_table::Vector{Float64}
    viterbi_backward::Vector{UInt, 3}

    # I can probably make this one non-mutable if I change the representation of these two a bit. Let's see.
    "This is L in the paper, i.e. the maximum length allowed for a word."
    max_word_length::UInt
    # I feel that this is just a convenience variable to preallocate array space so that even the longest sentence can be accomodated. I don't think the model itself has any sort of max length restriction on sentences.
    max_sentence_length::UInt

    function Sampler(npylm::NPYLM, max_word_length::UInt, max_sentence_length::UInt)
        sampler = new()
        sampler.npylm = npylm
        # TODO: Adapt this to the bigram case.
        sampler.word_ids = Vector{UInt}(undef, 3)
        # TODO: Need to write this function still.
        allocate_capacity(sampler, max_word_length, max_sentence_length)
        return sampler
    end
end

function allocate_capacity(sampler::Sampler, max_word_length::UInt, max_sentence_length::UInt)
    sampler.max_word_length = max_word_length
    sampler.max_sentence_length = max_sentence_length
    # Size of what
    size = max_sentence_length + 1
    sampler.log_z = Vector{Float64}(undef, size)
    sampler.scaling_coefficients = Vector{Float64}(undef, size)
    sampler.viterbi_backward = Array{Float64, 3}(undef, size, max_word_length+1, max_word_length+1)
    sampler.backward_sampling_table = Vector{Float64}(undef, max_word_length * max_word_length)

    # This is just convoluted. And also why do you need to + 1 after all. Is it just because of indexing? Quite nonsensical it seems to me.
    sampler.α_tensor = Array{Float64, 3}(0, size + 1, max_word_length + 1, max_word_length + 1)
    sampler.p_w_h = Array{Float64, 4}(0, size, max_word_length + 1, max_word_length + 1, max_word_length + 1)
    sampler.substring_word_id_cache = Array{UInt, 2}(0, size, max_word_length + 1)
end

"α[t][k][j] represents the marginal probability of string c1...ct with both the final k characters and further j preceding characters being words."
function get_substring_word_id_at_t_k(sampler::Sampler, sentence::Sentence, t::UInt, k::UInt)
    word_id = sampler.substring_word_id_cache[t][k]
    # 0 is used to indicate the initial state, where there's no cache.
    # Though wouldn't it conflict with BOS? Let's see then.
    if word_id == 0
        # In the Julia indexing system, all indices are to be shifted by 1.
        word_id = get_substr_word_id(sentence, t - k + 1, t)
        sampler.substring_word_id_cache[t][k] = word_id
    end
    return word_id
end

# If α[t-k][j][i] is already normalized, there's no need to normalize α[t][k][j]
function sum_α_t_k_j(sampler::Sampler, sentence::Sentence, t::UInt, k::UInt, j::UInt, prod_scaling::Float64)
    word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
    sentence_as_chars = sentence.characters
    length = length(sentence)
    # We need to generate <bos> in this case
    # TODO: I'm really unsatisfied with the constant manual generation of BOS and EOS. I mean why not generate it already when first reading in the corpus? This can probably save tons of problems.
    if j == 1
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = BOS
        sampler.word_ids[3] = word_k_id
        # Compute the probability of this word with length k
        p_w_h = compute_p_w(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, 2, t - k + 1, t)
        # I think the scaling is to make sure that this thing doesn't underflow.
        # I'm actually not even sure if this implementation is correct for the 3-gram case. Why doesn't he just focus on the bigram case first anyways.
        sampler.α_tensor[t][k][1] = p_w_h * prod_scaling
        sampler.p_w_h[t][k][1][1] = p_w_h
        return
    end
    # This is the same as where i = 0
    if t - k - j == 0
        word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = word_j_id
        sampler.word_ids[3] = word_k_id
        # Probably of the word with length k
        p_w_h = compute_p_w(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, 2, t - k + 1, t)
        # Should I append a (+ 1) here??? Damn it. The indices are so freaking confusing.
        sampler.α_tensor[t][k][j] = p_w_h * sampler.α_tensor[t - k][j][1] * prod_scaling
        sampler.p_w_h[t][k][j][1] = p_w_h
        return
    end
    # Perform marginalization in all other cases
    sum = 0.0
    
    for i in 1:min(t - k - j, sampler.max_word_length)
        word_i_id = 
    end
end