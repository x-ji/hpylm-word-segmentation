# include("NPYLM.jl")
# include("Corpus.jl")
# include("Sentence.jl")
# using OffsetArrays
"""
This structs holds all the necessary fields and functions for sampling sentence segmentations using forward-backward inference.
"""
mutable struct Sampler
    npylm::NPYLM
    "The word_ids of the current 3-gram being calculated"
    word_ids::OffsetVector{UInt}
    "Cache of ids of some words previously segmented in this sentence."
    # substring_word_id_cache::Array{Int, 2}
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
    log_z::OffsetVector{Float64}
    "Stores the inverse of the actual probabilities (one for each length). Used when `with_scaling`==true. The purpose is to combat probabilities that are way too low."
    scaling_coefficients::OffsetVector{Float64}
    """
    Table to temporarily hold possibilities for the sampling of j and k during the backward sampling.

    It's called a "table" because we first record the probabilities for all candidates j and k values. After that, we need to *draw* from this "table" actual j and k values.

    See line 8 of Figure 5 of the paper.
    """
    backward_sampling_table::OffsetVector{Float64}

    """
    Matrix to hold the indices of i that maximize the log probability of the trigram sequence.

    e.g. viterbi_backward_indices[t,k,j] = 2 means when the first gram (i) has length 2, the probability is maximized. This is why there isn't a `i` index, unlike the p_w_h_cache array
    """
    viterbi_backward_indices::OffsetArray{Int, 3}

    # I can probably make this one non-mutable if I change the representation of these two a bit. Let's see.
    "This is L in the paper, i.e. the maximum length allowed for a word."
    max_word_length::Int
    # I feel that this is just a convenience variable to preallocate array space so that even the longest sentence can be accomodated. I don't think the model itself has any sort of max length restriction on sentences.
    max_sentence_length::Int

    function Sampler(npylm::NPYLM, max_word_length::Int, max_sentence_length::Int)
        sampler = new()
        sampler.npylm = npylm
        # TODO: Adapt this to the bigram case.
        sampler.word_ids = OffsetVector{UInt}(undef, 0:2)
        allocate_capacity(sampler, max_word_length, max_sentence_length)
        return sampler
    end
end

# I'm not totally sure if this is still needed in Julia. Let's first see how it goes then.
function extend_capacity(sampler::Sampler, max_word_length::Int, max_sentence_length::Int)
    # If the existing capacity is already good enough then no need to extend
    if (max_word_length <= sampler.max_word_length && max_sentence_length <= sampler.max_sentence_length)
        return
    else
        allocate_capacity(sampler, max_word_length, max_sentence_length)
    end
end

function allocate_capacity(sampler::Sampler, max_word_length::Int, max_sentence_length::Int)
    sampler.max_word_length = max_word_length
    sampler.max_sentence_length = max_sentence_length
    # Size of arrays that contain the various values. Because we need to take care of situations involving BOS, the size of such arrays should be 1 longer than only the max_sentence_length
    size = max_sentence_length + 1
    sampler.log_z = OffsetVector{Float64}(undef, 0:max_sentence_length)
    # Why does this need to be one longer?
    sampler.scaling_coefficients = OffsetVector{Float64}(undef, 0:size)
    sampler.viterbi_backward_indices = OffsetArray{Int, 3}(undef, (0:max_sentence_length, 0:max_word_length, 0:max_word_length))
    sampler.backward_sampling_table = OffsetVector{Float64}(undef, 0:max_word_length * max_word_length - 1)

    # It needs to be 1 longer than length, because we have to accomodate for the index 0, which indicates that we have BOS as one of the grams.
    sampler.α_tensor = OffsetArray{Float64}(undef, 0:size, 0:max_word_length, 0:max_word_length)
    # sampler.p_w_h_cache = Array{Float64, 4}(undef, size, max_word_length + 1, max_word_length + 1, max_word_length + 1)
    sampler.p_w_h_cache = OffsetArray{Float64}(undef, 0:max_sentence_length, 0:max_word_length, 0:max_word_length, 0:max_word_length)
    # sampler.substring_word_id_cache = Array{Int, 2}(undef, size, max_word_length + 1)
    sampler.substring_word_id_cache = OffsetArray{UInt}(undef, 0:max_sentence_length, 0:max_word_length)
end

"""
α[t][k][j] represents the marginal probability of string c1...ct with both the final k characters and further j preceding characters being words.

This function returns the id of the word constituted by the last k characters of the total t characters.

Note that since this function already takes care of the index shift that's needed in Julia, the callers will still just call it normally.
"""
function get_substring_word_id_at_t_k(sampler::Sampler, sentence::Sentence, t::Int, k::Int)::UInt
    word_id = sampler.substring_word_id_cache[t,k]
    # 0 is used to indicate the initial state, where there's no cache.
    # Though wouldn't it conflict with BOS? Let's see then.
    if word_id == 0
        # Fuck the Julia indexing system. Let me stick to 0-based indexing for now.
        word_id = get_substr_word_id(sentence, t - k, t - 1)
        sampler.substring_word_id_cache[t,k] = word_id
    end
    return word_id
end

# If α[t-k][j][i] is already normalized, there's no need to normalize α[t][k][j]
raw"""
The step during forward filtering where α[t][k][j] is calculated

Note that in this trigram case, α[t][k][j] = \sum^{t-k-j}_{i=1} p(c^t_{t-k+1} | c^{t-k-j}_{t-k-j-i+1} c^{t-k}_{t-k-j+1}) * α[t - k][j][i]

That is to say, we first fix both the third gram and the second gram, and marginalize over different lengths of the first gram, indicated by the changing index i here.
"""
function calculate_α_t_k_j(sampler::Sampler, sentence::Sentence, t::Int, k::Int, j::Int, prod_scaling::Float64)
    word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
    sentence_as_chars = sentence.characters
    @assert(t <= sampler.max_sentence_length + 1)
    @assert(k <= sampler.max_word_length)
    @assert(j <= sampler.max_word_length)
    @assert(t - k >= 0)
    # I'm really unsatisfied with the constant manual generation of BOS and EOS. I mean why not generate it already when first reading in the corpus? This can probably save tons of problems.
    # However, I can now also see why this might be necessary: i.e. so that BOS and EOS, which are essentially special characters, might not be accidentally considered a part of a word when we try to determine word boundaries, which would result in nonsensical results... Um let's see. Let me first port the code anyways.

    # Speical case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
    if j == 0
        sampler.word_ids[0] = BOS
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = word_k_id
        # Compute the probability of this word with length k
        # Why do we - 1 in the end though?
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
        @assert p_w_h > 0.0
        # I think the scaling is to make sure that this thing doesn't underflow.
        # Store the values in the cache
        sampler.α_tensor[t,k,0] = p_w_h * prod_scaling
        sampler.p_w_h_cache[t,k,0,0] = p_w_h
        return
    # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
    elseif t - k - j == 0
        word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
        sampler.word_ids[0] = BOS
        sampler.word_ids[1] = word_j_id
        sampler.word_ids[2] = word_k_id
        # Probably of the word with length k, which is the last (3rd) word.
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
        @assert p_w_h > 0.0
        @assert sampler.α_tensor[t-k,j,0] > 0.0
        # In this case, the expression becomes the following.
        sampler.α_tensor[t,k,j] = p_w_h * sampler.α_tensor[t - k,j,0] * prod_scaling
        # The last index here is i. i == 0.
        sampler.p_w_h_cache[t,k,j,0] = p_w_h
        return
    else
        # Perform the normal marginalization procedure in all other cases
        sum = 0.0
        for i in 1:min(t - k - j, sampler.max_word_length)
            # Yeah I was most likely mixing indices up. All those indexing just don't make much sense let's go on and see then.
            word_i_id = get_substring_word_id_at_t_k(sampler, sentence, t - k - j, i)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            # The first gram
            sampler.word_ids[0] = word_i_id
            # The second gram
            sampler.word_ids[1] = word_j_id
            # The third gram
            sampler.word_ids[2] = word_k_id

            # This way of writing the code is still a bit messy. Let's see if we can do better then.
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
            @assert p_w_h > 0.0
            @assert i <= sampler.max_word_length
            @assert sampler.α_tensor[t-k,j,i] > 0.0
            # Store the word possibility in the cache tensor.
            sampler.p_w_h_cache[t,k,j,i] = p_w_h
            temp = p_w_h * sampler.α_tensor[t - k,j,i]
            sum += temp
        end
        @assert sum > 0.0
        sampler.α_tensor[t,k,j] = sum * prod_scaling
    end
end

function forward_filtering(sampler::Sampler, sentence::Sentence, with_scaling::Bool)
    sampler.α_tensor[0,0,0] = 1
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
            @assert sum_α > 0.0
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
    k = IntContainer(0)
    j = IntContainer(0)
    sum_length = 0
    backward_sample_k_and_j(sampler, sentence, t, 1, k, j)
    # I'm not sure yet why the segments array should contain their lengths instead of the word ids themselves. I guess it's just a way to increase efficiency and all that.
    segment_lengths = Int[]
    push!(segment_lengths, k.int)

    # There's only one word in total for the sentence.
    # Damn why did I return nothing here. This is dumb.
    if j.int == 0 && k.int == t
        return OffsetArray(reverse(segment_lengths), 0:length(segment_lengths) - 1)
    end

    @assert(k.int > 0 && j.int > 0)
    @assert(j.int <= sampler.max_word_length)

    push!(segment_lengths, j.int)
    t -= (k.int + j.int)
    sum_length += k.int + j.int
    next_word_length = j.int

    while (t > 0)
        # There's only ever one character left in the whole sentence
        if t == 1
            k.int = 1
            j.int = 0
        else
            backward_sample_k_and_j(sampler, sentence, t, next_word_length, k, j)
        end
        push!(segment_lengths, k.int)
        t -= k.int
        if j.int == 0
            # println("t is $(t)")
            @assert(t == 0)
        else
            @assert(j.int <= sampler.max_word_length)
            push!(segment_lengths, j.int)
            t -= j.int
        end
        sum_length += (k.int + j.int)
        next_word_length = j.int
    end
    @assert(t == 0)
    @assert(sum_length == length(sentence))
    # result = OffsetArray(reverse(segment_lengths), 0:length(segment_lengths) - 1)
    # println("In backward_sampling, sentence is $sentence, segment_lengths is $segment_lengths, result is $result")
    return OffsetArray(reverse(segment_lengths), 0:length(segment_lengths) - 1)
end

"""
Returns k and j in a tuple

"next_word" really means the target word, the last gram in the 3 gram, e.g. the EOS in p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
"""
function backward_sample_k_and_j(sampler::Sampler, sentence::Sentence, t::Int, third_gram_length::Int, sampled_k::IntContainer, sampled_j::IntContainer)
    table_index = 0
    sentence_as_chars = sentence.characters
    sentence_length = length(sentence)
    sum_p = 0.0
    for k in 1:min(t, sampler.max_word_length)
        for j in 1:min(t - k, sampler.max_word_length)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
            # When we begin the backward sampling on a sentence, note that the final token is always EOS. We have probabilities p(EOS | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            word_t_id = EOS
            if t < length(sentence)
                @assert(t + third_gram_length <= length(sentence))
                @assert(third_gram_length > 0)
                # Otherwise the final token is not EOS already but an actual word. Still the principles for sampling don't change.
                word_t_id = get_substring_word_id_at_t_k(sampler, sentence, t + third_gram_length, third_gram_length)
            end
            sampler.word_ids[0] = word_j_id
            sampler.word_ids[1] = word_k_id
            sampler.word_ids[2] = word_t_id
            p_w_h = 0.0
            if t == length(sentence)
                # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t, t)
            else
                # We should have already cached this value.
                p_w_h = sampler.p_w_h_cache[t + third_gram_length, third_gram_length, k, j]
            end
            @assert(sampler.α_tensor[t,k,j] > 0)
            # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            p = p_w_h * sampler.α_tensor[t,k,j]
            @assert(p > 0)
            sampler.backward_sampling_table[table_index] = p
            sum_p += p
            # println("p_w_h is $(p_w_h), sampler.α_tensor[t,k,j] is $(sampler.α_tensor[t,k,j]), p is $(p), sum_p is $(sum_p)")
            table_index += 1
        end
        # TODO: One can likely refactor this code a bit more. Just setting j = 0 should be enough shouldn't it?
        # In this case the first gram is BOS. The third gram is EOS.
        if t == k
            j = 0
            word_j_id = BOS
            word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
            word_t_id = EOS
            if t < length(sentence)
                @assert(t + third_gram_length <= length(sentence))
                @assert(third_gram_length > 0)
                word_t_id = get_substring_word_id_at_t_k(sampler, sentence, t + third_gram_length, third_gram_length)
            end
            sampler.word_ids[0] = word_j_id
            sampler.word_ids[1] = word_k_id
            sampler.word_ids[2] = word_t_id
            p_w_h = 0.0
            if t == length(sentence)
                # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t, t)
            else
                # We should have already cached this value.
                p_w_h = sampler.p_w_h_cache[t + third_gram_length, third_gram_length, k, j]
            end
            @assert(sampler.α_tensor[t,k,j] > 0)
            # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            p = p_w_h * sampler.α_tensor[t,k,j]
            @assert(p > 0)
            sampler.backward_sampling_table[table_index] = p
            sum_p += p
            table_index += 1
        end
    end

	@assert(table_index > 0);
	@assert(table_index <= sampler.max_word_length * sampler.max_word_length);

    # Eventually, the table should have (min(t, sampler.max_word_length) * min(t - k, sampler.max_word_length)) + 1 entries
    # This is such a pain. We should definitely be able to simplify the code much more than this. Eh.
    normalizer = 1.0 / sum_p
    # println("Normalizer is $(normalizer), sum_p is $(sum_p)")
    randnum = rand(Float64)
    index = 0
    stack = 0.0
    for k in 1:min(t, sampler.max_word_length)
        for j in 1:min(t - k, sampler.max_word_length)
            @assert(index < table_index)
            @assert(sampler.backward_sampling_table[index] > 0.0)
            stack += sampler.backward_sampling_table[index] * normalizer
            # println("randnum is $(randnum), stack is $(stack)")
            if randnum < stack
                sampled_k.int = k
                sampled_j.int = j
                return
            end
            index += 1
        end

        # The special case where the first gram is BOS. The last entry of the table.
        if t == k
            # println("t == k triggered!")
            @assert(index < table_index)
            @assert(sampler.backward_sampling_table[index] > 0.0)
            stack += sampler.backward_sampling_table[index] * normalizer
            if randnum < stack
                sampled_k.int = k
                sampled_j.int = 0
                return
            end
            index += 1
        end
    end
    # Sometimes this can somehow fall through?
    println("Fell through!")
end

# TODO: This function name is not totally accurate as "blocked Gibbs" is really the name of the whole procedure, while this function only takes care of the "draw segmentation" part.
"Does the segment part in the blocked Gibbs algorithm (line 6 of Figure 3 of the paper)"
function blocked_gibbs_segment(sampler::Sampler, sentence::Sentence, with_scaling::Bool)
    for i in 0:length(sentence)
        for j in 0:sampler.max_word_length
            sampler.substring_word_id_cache[i,j] = 0
        end
    end

    forward_filtering(sampler, sentence, with_scaling)
    return backward_sampling(sampler, sentence)
end

# For the 3-gram case, we need to use viterbi decoding to eventually produce the most likely sequence of segments.
# TODO: OK so what's the difference between the viterbi methods and the original methods without viterbi? I'm kinda lost again. Guess I'll first have to look through the whole training and evaluation flows to look for clues then. Let's see.
function viterbi_argmax_calculate_α_t_k_j(sampler::Sampler, sentence::Sentence, t::Int, k::Int, j::Int)
    word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
    sentence_as_chars = sentence.characters
	@assert(t <= sampler.max_sentence_length + 1);
	@assert(k <= sampler.max_word_length);
	@assert(j <= sampler.max_word_length);
    # Special case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
    if j == 0
        sampler.word_ids[0] = BOS
        sampler.word_ids[1] = BOS
        sampler.word_ids[2] = word_k_id
        # Compute the probability of this word with length k
        # Why do we - 1 in the end though? We probably shouldn't do so here since the indexing system is different. Eh.
        # TODO: Just use 0-based index anyways if things don't work out.
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
        @assert(p_w_h > 0.0)
        # I think the scaling is to make sure that this thing doesn't underflow.
        # Store the values in the cache
        sampler.α_tensor[t,k,0] = log(p_w_h)
        sampler.viterbi_backward_indices[t,k,0] = 0
        return
    # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
    elseif t - k - j == 0
        word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
        sampler.word_ids[0] = BOS
        sampler.word_ids[1] = word_j_id
        sampler.word_ids[2] = word_k_id
        # Probability of the word with length k, which is the last (3rd) word.
        p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
        @assert(p_w_h > 0.0)
        @assert(sampler.α_tensor[t-k,j,0] != 0.0)
        # In this case, the expression becomes the following.
        sampler.α_tensor[t,k,j] = log(p_w_h) + sampler.α_tensor[t - k,j,0]
        sampler.viterbi_backward_indices[t,k,j] = 0
        return
    else
        # Perform the normal marginalization procedure in all other cases
        max_log_p = 0.0
        argmax = 0
        for i in 1:min(t - k - j, sampler.max_word_length)
            word_i_id = get_substring_word_id_at_t_k(sampler, sentence, t - k - j, i)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            # The first gram
            sampler.word_ids[0] = word_i_id
            # The second gram
            sampler.word_ids[1] = word_j_id
            # The third gram
            sampler.word_ids[2] = word_k_id

            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t - k, t - 1)
            @assert(p_w_h > 0.0)
            @assert(i <= sampler.max_word_length)
            # Because it's a log value then.
            @assert(sampler.α_tensor[t-k,j,i] <= 0)
            temp = log(p_w_h) + sampler.α_tensor[t - k,j,i]
            @assert(temp <= 0)

            # We're trying to determine the i value (first gram) that maximizes the possibility
            if (argmax == 0 || temp > max_log_p)
                argmax = i
                max_log_p = temp
            end
        end
        @assert(argmax > 0)
        sampler.α_tensor[t,k,j] = max_log_p
        # We use the viterbi_backward_indices matrix to store the i value that maximizes the possibility of the trigram.
        sampler.viterbi_backward_indices[t,k,j] = argmax
    end
end

function viterbi_forward_filtering(sampler::Sampler, sentence::Sentence)
    for t in 1:length(sentence)
        for k in 1:min(t, sampler.max_word_length)
            # There is no j, i.e. the second gram is also BOS.
            if t == k
                viterbi_argmax_calculate_α_t_k_j(sampler, sentence, t, k, 0)
            end
            # Note that in the t==k case, we will have range 1:0 which is automatically empty, so the following code will not be run.
            for j in 1:min(t - k, sampler.max_word_length)
                viterbi_argmax_calculate_α_t_k_j(sampler, sentence, t, k, j)
            end
        end
    end
end

"This method is called when we know the third gram is EOS, so we're only sampling the first gram and second gram."
function viterbi_argmax_backward_sample_k_and_j_to_eos(sampler::Sampler, sentence::Sentence, t::Int, third_gram_length::Int, argmax_k::IntContainer, argmax_j::IntContainer)
    @assert(t == length(sentence))
    table_index = 0
    sentence_as_chars = sentence.characters
    sentence_length = length(sentence)
    max_log_p = 0.0
    argmax_k.int = 0
    argmax_j.int = 0
    for k in 1:min(t, sampler.max_word_length)
        for j in 1:min(t - k, sampler.max_word_length)
            word_j_id = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
            # When we begin the backward sampling on a sentence, note that the final token is always EOS. We have probabilities p(EOS | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            sampler.word_ids[0] = word_j_id
            sampler.word_ids[1] = word_k_id
            sampler.word_ids[2] = EOS
            # It's still the EOS. We just wrote it in a simpler way.
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t, t)
            @assert sampler.α_tensor[t,k,j] <= 0
            temp = log(p_w_h) + sampler.α_tensor[t,k,j]
            @assert temp <= 0
            if (argmax_k.int == 0 || temp > max_log_p)
                max_log_p = temp
                argmax_k.int = k
                argmax_j.int = j
            end
        end
        # TODO: One can likely refactor this code a bit more.
        # In this case the first gram is BOS. The third gram is EOS.
        if t == k
            word_j_id = BOS
            word_k_id = get_substring_word_id_at_t_k(sampler, sentence, t, k)
            word_t_id = EOS
            # We removed all code regarding cases where t < length of the original sentence. I think this is because we always know that this will be the case where t == length(sentence) and that the third gram will always be the EOS, i.e. this method will only be called in such cases.
            # So apparently we can simplify the code a bit and maybe put it together with another method eh?
            sampler.word_ids[0] = word_j_id
            sampler.word_ids[1] = word_k_id
            sampler.word_ids[2] = word_t_id
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t, t)
            @assert sampler.α_tensor[t,k,0] <= 0
            # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
            temp = log(p_w_h) + sampler.α_tensor[t,k,0]
            @assert temp <= 0
            if argmax_k.int == 0 || temp > max_log_p
                max_log_p = temp
                argmax_k.int = k
                argmax_j.int = 0
            end
        end
    end
end

function viterbi_backward_sampling(sampler::Sampler, sentence::Sentence)
    segment_lengths = Int[]
    t = length(sentence)
    sum_length = 0
    kc = IntContainer(0)
    jc = IntContainer(0)
    viterbi_argmax_backward_sample_k_and_j_to_eos(sampler, sentence, t, 1, kc, jc)
    k = kc.int
    j = jc.int
    @assert(k <= sampler.max_word_length)
    # I'm not sure yet why the segments array should contain their lengths instead of the word ids themselves. I guess it's just a way to increase efficiency and all that.
    push!(segment_lengths, k)
    sum_length += k

    # There's only one word in total for the sentence.
    if j == 0 && k == t
        return
    end

    @assert(k > 0 && j > 0)
    @assert(j <= sampler.max_word_length)
    push!(segment_lengths, j)
    # We knew that i is the index that maximizes the possibility of the trigram
    i = sampler.viterbi_backward_indices[t,k,j]
    @assert i >= 0
    @assert i <= sampler.max_word_length
    sum_length += j + i

    # Move the "sentence end" forward
    t -= k
    k = j
    j = i

    # The sentence is already fully segmented.
    if i == 0
        @assert sum_length == length(sentence)
        return
    end

    push!(segment_lengths, i)

    # TODO: This code repetition surely is also avoidable?
    # Repeatedly push forward the end, taking advantage of the already recorded viterbi indices.
    while (t > 0)
        i = sampler.viterbi_backward_indices[t,k,j]
        @assert i >= 0
        @assert i <= sampler.max_word_length
        if (i != 0)
            push!(segment_lengths, i)
        end
        t -= k
        k = j
        j = i
        sum_length += i
    end
    @assert(t == 0)
    @assert(sum_length == length(sentence))
    @assert length(segment_lengths) > 0
    return OffsetArray(reverse(segment_lengths), 0:length(segment_lengths) - 1)
end

"This function uses viterbi algorithm to sample the segmentation of a sentence, instead of the approach in the `blocked_gibbs_segment` function above. They should both be valid approaches."
function viterbi_decode(sampler::Sampler, sentence::Sentence)
    # array_length = length(sentence) + 1
    sampler.α_tensor[0,0,0] = 0.0
    sampler.log_z[0] = 0.0
    for t in 0:length(sentence)
        for k in 0:sampler.max_word_length
            sampler.substring_word_id_cache[t,k] = 0
        end
    end
    viterbi_forward_filtering(sampler, sentence)
    return viterbi_backward_sampling(sampler, sentence)
end

"Computes the probability of resulting in EOS with the given α_tensor for the sentence."
function compute_log_forward_probability(sampler::Sampler, sentence::Sentence, with_scaling::Bool)
    enumerate_forward_variables(sampler, sentence, with_scaling)
    # It should now point to EOS
    t = length(sentence) + 1
    if !with_scaling
        # Consider the length of EOS to be 1
        k = 1
        α_eos = 0.0
        # As described in the paper, we need to sum all possible previous permutations together in this case.
        for j in 1:min(t - k, sampler.max_word_length)
            @assert(sampler.α_tensor[t,k,j] > 0.0)
            α_eos += sampler.α_tensor[t,k,j]
        end
        @assert(α_eos > 0.0)
        return log(α_eos)
    else
        # If we use scaling, we stored the scaling coefficients as the inverse of the actual probabilities.
        log_p_x = 0.0
        for i in 1:t
            log_p_x += log(1.0 / sampler.scaling_coefficients[i])
        end
        return log_p_x
    end
end

# TODO: This function is a duplicate of some of the functionalities that we already performed above. Should be able to put it somewhere.
function enumerate_forward_variables(sampler::Sampler, sentence::Sentence, with_scaling::Bool)
    # array_length = length(sentence) + 1
    for i in 0:length(sentence)
        for j in 0:sampler.max_word_length
            sampler.substring_word_id_cache[i,j] = 0
        end
    end

    sentence_as_chars = sentence.characters
    # This should fill the alpha tensor before thef final EOS.
    forward_filtering(sampler, sentence, with_scaling)
    # Calculate the possibility of producing EOS as the final gram.
    # Though isn't EOS also a part of the original sentence? Doesn't seem to be the case then due to the special processing I guess. Still a bit confusing that's for sure... Let's just go on and do better then. Eh.
    α_eos = 0.0
    t = length(sentence) + 1
    k = 1
    for j in 1:min(t - k, sampler.max_word_length)
        prob_sum = 0.0
        for i in ((t - k - j == 0) ? 0 : 1):min(t - k - j, sampler.max_word_length)
            sampler.word_ids[0] = get_substring_word_id_at_t_k(sampler, sentence, t - k - j, i)
            sampler.word_ids[1] = get_substring_word_id_at_t_k(sampler, sentence, t - k, j)
            sampler.word_ids[2] = EOS
            p_w_h = compute_p_w_of_nth_word(sampler.npylm, sentence_as_chars, sampler.word_ids, 2, t, t)
            @assert(p_w_h > 0.0)
            prob_sum += p_w_h * sampler.α_tensor[t-k,j,i]
        end
        sampler.α_tensor[t,k,j] = prob_sum
        α_eos += prob_sum
    end
    if with_scaling
        sampler.scaling_coefficients[t] = 1.0 / α_eos
    end
end
