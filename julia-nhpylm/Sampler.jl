include("NPYLM.jl")
using OffsetArray
"""
This structs holds all the necessary fields and functions for sampling sentence segmentations using forward-backward inference.
"""
mutable struct Sampler
    npylm::NPYLM
    "The word_ids of the current 3-gram being calculated"
    word_ids::Vector{UInt}
    "Cache of ids of some words previously segmented in this sentence."
    # substring_word_id_cache::Array{UInt, 2}
    substring_word_id_cache::OffsetArray{UInt}
    "3-dimensional tensor that contains the forward variables, i.e. in α[t][k][j] at p.104 of the paper"
    # α_tensor::Array{Float64, 3}
    α_tensor::OffsetArray{Float64}
    """
    Cache of word probabilities p_w given by the CHPYLM. Caching the value is useful so that we can avoid excessive repeated computations.
    
    I think the "h" here actually stands for Theta or something

    Indexing: p_w_h_cache[t][k][j][i]
    - t: The full length of the sentence
    - k: The length of the third gram, the last word
    - j: The length of the second gram
    - i: The length of the first gram
    """
    # One problem is that we can't use zero indexing. So in cases where the first gram is BOS we still need to do something else.
    # Still, as I suggested, why don't we just make BOS inherently a part of the sentence when we read it in. Is there any problem with that?
    # p_w_h_cache::Array{Float64, 4}
    p_w_h_cache::OffsetArray{Float64}
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

    # The + 1 part is because we have to accomodate for the index 0, which indicates that we have BOS as one of the grams.
    # sampler.α_tensor = Array{Float64, 3}(undef, size + 1, max_word_length + 1, max_word_length + 1)
    sampler.α_tensor = OffsetArray{Float64}(undef, 0:size + 1, max_word_length + 1, max_word_length + 1)
    # sampler.p_w_h_cache = Array{Float64, 4}(undef, size, max_word_length + 1, max_word_length + 1, max_word_length + 1)
    sampler.p_w_h_cache = OffsetArray{Float64}(undef, 0:size - 1, 0:max_word_length, 0:max_word_length, 0:max_word_length)
    # sampler.substring_word_id_cache = Array{UInt, 2}(undef, size, max_word_length + 1)
    sampler.substring_word_id_cache = OffsetArray{UInt}(undef, 0:size-1, 0:max_word_length)
end

"""
α[t][k][j] represents the marginal probability of string c1...ct with both the final k characters and further j preceding characters being words.

This function returns the id of the word constituted by the last k characters of the total t characters.

Note that since this function already takes care of the index shift that's needed in Julia, the callers will still just call it normally.
"""
function get_substring_word_id_at_t_k(sampler::Sampler, sentence::Sentence, t::UInt, k::UInt)::UInt
    word_id = sampler.substring_word_id_cache[t,k]
    # 0 is used to indicate the initial state, where there's no cache.
    # Though wouldn't it conflict with BOS? Let's see then.
    if word_id == 0
        # In the Julia indexing system, all indices are to be shifted by 1.
        # e.g. the last 3 characters in a length 5 string will be: s[5-3+1:5]
        word_id = get_substr_word_id(sentence, t - k + 1, t)
        sampler.substring_word_id_cache[t,k] = word_id
    end
    return word_id
end

# If α[t-k][j][i] is already normalized, there's no need to normalize α[t][k][j]
"""raw
The step during forward filtering where α[t][k][j] is calculated

Note that in this trigram case, α[t][k][j] = \sum^{t-k-j}_{i=1} p(c^t_{t-k+1} | c^{t-k-j}_{t-k-j-i+1} c^{t-k}_{t-k-j+1}) * α[t - k][j][i]

That is to say, we first fix both the third gram and the second gram, and marginalize over different lengths of the first gram, indicated by the changing index i here.
"""
function calculate_α_t_k_j(sampler::Sampler, sentence::Sentence, t::UInt, k::UInt, j::UInt, prod_scaling::Float64)
    word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
    sentence_as_chars = sentence.characters
    # I'm really unsatisfied with the constant manual generation of BOS and EOS. I mean why not generate it already when first reading in the corpus? This can probably save tons of problems.
    # However, I can now also see why this might be necessary: i.e. so that BOS and EOS, which are essentially special characters, might not be accidentally considered a part of a word when we try to determine word boundaries, which would result in nonsensical results... Um let's see. Let me first port the code anyways.

    # Speical case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
    if j == 0
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = BOS
        sampler.word_ids[3] = word_k_id
        # Compute the probability of this word with length k
        # Why do we - 1 in the end though?
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
        # I think the scaling is to make sure that this thing doesn't underflow.
        # Store the values in the cache
        sampler.α_tensor[t,k,0] = p_w_h * prod_scaling
        sampler.p_w_h_cache[t,k,0,0] = p_w_h

    # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
    elseif t - k - j == 0
        word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = word_j_id
        sampler.word_ids[3] = word_k_id
        # Probably of the word with length k, which is the last (3rd) word.
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
        # In this case, the expression becomes the following.
        sampler.α_tensor[t,k,j] = p_w_h * sampler.α_tensor[t - k,j,0] * prod_scaling
        # The last index here is i. i == 0.
        sampler.p_w_h_cache[t,k,j,0] = p_w_h
    else
        # Perform the normal marginalization procedure in all other cases
        sum = 0.0
        for i in 1:min(t - k - j, sampler.max_word_length)
            word_i_id = get_substring_word_id_at_t_k(sampler, sentence, t - k - j, i)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            # The first gram
            sampler.word_ids[1] = word_i_id
            # The second gram
            sampler.word_ids[2] = word_j_id
            # The third gram
            sampler.word_ids[3] = word_k_id

            # This way of writing the code is still a bit messy. Let's see if we can do better then.
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
            # Store the word possibility in the cache tensor.
            sampler.p_w_h_cache[t,k,j,i] = p_w_h

            temp = p_w_h * sampler.α_tensor[t - k,j,i]
            sum += temp
        end
        sampler.α_tensor[t,k,j] = sum * prod_scaling
    end
end

function forward_filtering(sampler::Sampler, sentence::Sentence, with_scaling::Bool)
    sampler.α_array[0,0,0] = 1
    for t in 1:length(sentence)
        prod_scaling = 1.0
        # The original paper most likely made a mistake on this. Apparently one should take min(t, L) instead of max(1, t - L) which makes no sense.
        for k in 1:min(t, sampler.max_word_length)
            if (with_scaling && k > 1)
                # TODO: Why + 1 though. Need to understand prod_scaling better.
                prod_scaling *= sampler.scaling_coefficients[t - k + 1]
            end
            # If t - k = 0 then the loop will not be executed at all.
            for j in (t == k ? 0 : 1):min(t - k, sampler.max_word_length)
                sampler.α_tensor[t,k,j] = 0
                calculate_α_t_k_j(sampler, sentence, t, k, j, prod_scaling)
            end
        end

        # Perform scaling operations on the alpha tensor, in order to avoid underflowing.
        if (with_scaling)
            sum_α = 0.0
            for k in 1:min(t, sampler.max_word_length)
                for j in (t == k ? 0 : 1):min(t - k, sampler.max_word_length)
                    sum_α += sampler.α_tensor[t,k,j]
                end
            end
            sampler.scaling_coefficients[t] = 1.0 / sum_α
            for k in 1:min(t, sampler.max_word_length)
                for j in (t == k ? 0 : 1):min(t - k, sampler.max_word_length)
                    sampler.α_tensor[t,k,j] *= sampler.scaling_coefficients[t]
                end
            end
        end
    end
end

function backward_sampling(sampler::Sampler, sentence::Sentence)
    t = length(sentence)
    sum_length = 0
    (k, j) = backward_sample_k_and_j(sampler, sentence, t, 1)
    # I'm not sure yet why the segments array should contain their lengths instead of the word ids themselves. I guess it's just a way to increase efficiency and all that.
    segment_lengths = UInt[]
    push!(segment_lengths, k)

    # There's only one word in total for the sentence.
    if j == 0 && k == t
        return
    end

    push!(segment_lengths, j)
    t -= (k + j)
    sum_length += k + j
    next_word_length = j

    while (t > 0)
        # There's only ever one character left in the whole sentence
        if t == 1
            k = 1
            j = 0
        else
            (k, j) = backward_sample_k_and_j(sampler, sentence, t, next_word_length)
        end
        push!(segment_lengths, k)
        t -= k
        if j == 0
            @assert(t == 0)
        else
            push!(segment_lengths, j)
            t -= j
        end
        sum_length += (k + j)
        next_word_length = j
    end
    @assert(t == 0)
    @assert(sum_length == length(sentence))
    return reverse(segment_lengths)
end

"""
Returns k and j in a tuple

"next_word" really means the target word, the last gram in the 3 gram, e.g. the EOS in p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
"""
function backward_sample_k_and_j(sampler::Sampler, sentence::Sentence, t::UInt, third_gram_length::UInt)::Tuple{UInt,UInt}
    # Indices start from 1
    table_index = 1
    sentence_as_chars = sentence.characters
    sentence_length = length(sentence)
    sum_p = 0.0
    for k in 1:min(t, sampler.max_word_length)
        for j in 1:min(t - k, sampler.max_word_length)
            word_j_id = get_substring_word_id_at_t_k(sentence, t - k, j)
            word_k_id = get_substring_word_id_at_t_k(sentence, t, k)
            # When we begin the backward sampling on a sentence, note that the final token is always EOS. We have probabilities p(EOS | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            word_t_id = EOS
            if t < length(sentence)
                # Otherwise the final token is not EOS already but an actual word. Still the principles for sampling don't change.
                word_t_id = get_substring_word_id_at_t_k(sentence, t + third_gram_length, third_gram_length)
            end
            sampler.word_ids[1] = word_j_id
            sampler.word_ids[2] = word_k_id
            sampler.word_ids[3] = word_t_id
            p_w_h = 0.0
            if t == length(sentence)
                # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t, t)
            else
                # We should have already cached this value.
                p_w_h = sampler.p_w_h_cache_cache[t + third_gram_length, third_gram_length, k, j]
            end
            # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            p = p_w_h * sampler.α_tensor[t,k,j]
            sampler.backward_sampling_table[table_index] = p
            sum_p += p
            table_index += 1
        end
        # TODO: One can likely refactor this code a bit more.
        # In this case the first gram is BOS
        if t == k
            j = 0
            word_j_id = BOS
            word_k_id = get_substring_word_id_at_t_k(sentence, t, k)
            word_t_id = EOS
            if t < length(sentence)
                word_t_id = get_substring_word_id_at_t_k(sentence, t + third_gram_length, third_gram_length)
            end
            sampler.word_ids[1] = word_j_id
            sampler.word_ids[2] = word_k_id
            sampler.word_ids[3] = word_t_id
            p_w_h = 0.0
            if t == length(sentence)
                # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t, t)
            else
                # We should have already cached this value.
                p_w_h = sampler.p_w_h_cache_cache[t + third_gram_length, third_gram_length, k, j]
            end
            # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            p = p_w_h * sampler.α_tensor[t,k,j]
            sampler.backward_sampling_table[table_index] = p
            sum_p += p
            table_index += 1
        end
    end
    
    # Eventually, the table should have (min(t, sampler.max_word_length) * min(t - k, sampler.max_word_length)) + 1 entries
    # This is such a pain. We should definitely be able to simplify the code much more than this. Eh.
    normalizer = 1.0 / sum_p
    randnum = rand(Float64)
    index = 1
    stack = 0.0
    for k in 1:min(t, sampler.max_word_length)
        for j in 1:min(t - k, sampler.max_word_length)
            stack += sampler.backward_sampling_table[index] * normalizer
            if randnum < stack
                return (k, j)
            end
            index += 1
        end

        # The special case where the first gram is BOS. The last entry of the table.
        if t == k
            stack += sampler.backward_sampling_table[index] * normalizer
            if randnum < stack
                return (k, 0)
            end
            index += 1
        end
    end
end

function blocked_gibbs(sampler::Sampler, sentence::Sentence, segment_lengths::Vector{UInt}, with_scaling::Bool)
    array_length = length(sentence) + 1

    for i in 0:array_length
        for j in 0:sampler.max_word_length + 1
            sampler.substring_word_id_cache[i,j] = 0
        end
    end

    forward_filtering(sampler, sentence, with_scaling)
    backward_sampling(sampler, sentence, segment_lengths)
end

# For the 3-gram case, we need to use viterbi decoding to eventually produce the most likely sequence of segments.
# TODO: OK so what's the difference between the viterbi methods and the original methods without viterbi? I'm kinda lost again. Guess I'll first have to look through the whole training and evaluation flows to look for clues then. Let's see.
function viterbi_argmax_calculate_α_t_k_j(sampler::Sampler, sentence::Sentence, t::UInt, k::UInt, j::UInt)
    word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
    sentence_as_chars = sentence.characters
    # Special case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
    if j == 0
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = BOS
        sampler.word_ids[3] = word_k_id
        # Compute the probability of this word with length k
        # Why do we - 1 in the end though?
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
        # I think the scaling is to make sure that this thing doesn't underflow.
        # Store the values in the cache
        sampler.α_tensor[t,k,0] = log(p_w_h)
        sampler.p_w_h_cache[t,k,0,0] = p_w_h

    # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
    elseif t - k - j == 0
        word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = word_j_id
        sampler.word_ids[3] = word_k_id
        # Probably of the word with length k, which is the last (3rd) word.
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
        # In this case, the expression becomes the following.
        sampler.α_tensor[t,k,j] = p_w_h * sampler.α_tensor[t - k,j,0] * prod_scaling
        # The last index here is i. i == 0.
        sampler.p_w_h_cache[t,k,j,0] = p_w_h
    else
        # Perform the normal marginalization procedure in all other cases
        sum = 0.0
        for i in 1:min(t - k - j, sampler.max_word_length)
            word_i_id = get_substring_word_id_at_t_k(sampler, sentence, t - k - j, i)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            # The first gram
            sampler.word_ids[1] = word_i_id
            # The second gram
            sampler.word_ids[2] = word_j_id
            # The third gram
            sampler.word_ids[3] = word_k_id

            # This way of writing the code is still a bit messy. Let's see if we can do better then.
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 3, t - k + 1, t)
            # Store the word possibility in the cache tensor.
            sampler.p_w_h_cache[t,k,j,i] = p_w_h

            temp = p_w_h * sampler.α_tensor[t - k,j,i]
            sum += temp
        end
        sampler.α_tensor[t,k,j] = sum * prod_scaling
    end
end