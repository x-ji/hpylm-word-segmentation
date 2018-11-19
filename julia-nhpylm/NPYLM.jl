include("Def.jl")
include("Corpus.jl")
include("PYP.jl")
include("HPYLM.jl")
include("WHPYLM.jl")
include("CHPYLM.jl")
using Distributions

mutable struct NPYLM
    "The hierarhical Pitman-Yor model for words"
    whpylm::WHPYLM
    "The hierarhical Pitman-Yor model for characters"
    chpylm::CHPYLM

    # I think it's the case that each word has multiple corresponding tables, with each table existing at a different depth. Let's see then.
    # Each entry is a tablegroup for this dish actually. I think it's just the case that depths[1] means the tablegroup for this dish at depth 0 of this model. depths[2] means the tablegroup for this dish at depth 1 of this model, etc.
    # Still it doesn't seem correct. If it's "removed_from_table_k" then what does it have to do with depths? Let's see further then.
    # Yeah whatever the depth here is definitely something related to the inf Markov model. We still need to see a bit further though.
    prev_depth_at_table_of_token::Dict{UInt, Vector{Vector{Int}}}

    "The cache of WHPYLM G_0. This will be invalidated once the seating arrangements in the CHPYLM change."
    whpylm_G_0_cache::Dict{UInt, Float64}
    "The cache of CHPYLM G_0"
    chpylm_G_0_cache::Dict{UInt, Float64}
    λ_for_types::Vector{Float64}
    "Probability of generating a word of length k from the CHPYLM"
    chpylm_p_k::Vector{Float64}
    max_word_length::UInt
    max_sentence_length::UInt
    """
    The shape parameter of the Gamma distribution for estimating the λ value for the Poisson distribution. (Expression (16) of the paper)

    The relation: λ ~ Ga(a, b), where a is the shape and b is rate (not scale)
    """
    λ_a::Float64

    """
    The rate (not scale) parameter of the Gamma distribution for estimating the λ value for the Poisson distribution. (Expression (16) of the paper)

    The relation: λ ~ Ga(a, b), where a is the shape and b is rate (not scale)
    """
    λ_b::Float64
    "Cache for easier computation"
    whpylm_parent_p_w_cache::Vector{Float64}
    """
    Cache for the characters that make up the sentence that was last added to the chpylm

    Note that because this is a container that gets reused over and over again, its length is simply the maximum word length. The "actual word length" will be computed and passed in as a parameter to functions that use this variable. Not sure if this is the most sensible approach though. Maybe we can refactor it to a less tedious way later. The extra length parameter is no fun.
    """
    most_recent_word_added_to_chpylm::Vector{Char}

    function NPYLM(max_word_length::UInt, max_sentence_length::UInt, G_0::Float64, initial_λ_a::Float64, initial_λ_b::Float64, chpylm_beta_stop::Float64, chpylm_beta_pass::Float64)
        npylm = new()

        whpylm = WHPYLM(3)
        chpylm = CHPYLM(G_0, max_sentence_length, chpylm_beta_stop, chpylm_beta_pass)

        # Is this really what the original paper proposed? I feel like the author is probably overcomplicating stuffs. Let's see then. For now let me just use one poisson distribution for one iteration of training anyways.
        # TODO: Expand upon word types and use different poisson distributions for different types.
        npylm.λ_for_types = zeros(Float64, NUM_WORD_TYPES)
        # Currently we use a three-gram model.
        npylm.whpylm_parent_p_w_cache = zeros(Float64, 3)
        set_λ_prior(npylm, initial_λ_a, initial_λ_b)

        npylm.max_sentence_length = max_sentence_length
        npylm.max_word_length = max_word_length
        # + 2 because of bow and eow
        # Not sure if this is the most sensible approach with Julia. Surely we can adjust for that.
        npylm.most_recent_word_added_to_chpylm = Vector{Char}(undef, max_sentence_length + 2)
        # There are two extra cases where k = 1 and k > max_word_length
        npylm.chpylm_p_k = Vector{Float64}(undef, max_word_length + 2)
        for k in 1:max_word_length + 2
            npylm.chpylm_p_k[k] = 1.0 / (max_word_length + 2)
        end
        return npylm
    end
end

function produce_word_with_bow_and_eow(sentence_as_chars::Vector{Char}, word_begin_index::UInt, word_end_index::UInt, word::Vector{Char})
    word[1] = BOW
    # # The length is end - begin + 1. This is always the case.
    # for i in 1:word_end_index - word_begin_index + 1
    #     # - 1 because Julia arrays are 1-indexed
    #     word[i + 1] = sentence[word_begin_index + i - 1]
    # end
    word[2:2 + word_end_index - word_begin_index] = sentence_as_chars[word_begin_index:word_end_index]
    # One past the end of the word.
    word[word_end_index - word_begin_index + 1 + 1] = EOW
end


function reserve(npylm::NPYLM, max_sentence_length::UInt)
    if (max_sentence_length <= npylm.max_sentence_length)
        return
    end
    allocate_capacity(npylm, max_sentence_length)
end

function allocate_capacity(npylm::NPYLM, max_sentence_length::UInt)
    npylm.max_sentence_length = max_sentence_length
    npylm.most_recent_word_added_to_chpylm = Vector{Char}(max_sentence_length + 2)
end

# TODO: Probably can do without this one
function set_chpylm_G_0(npylm::NPYLM, G_0::Float64)
    set_G_0(npylm.chpylm, G_0)
end

function set_λ_prior(npylm::NPYLM, a::Float64, b::Float64)
    npylm.λ_a = a
    npylm.λ_b = b
    sample_λ_with_initial_params(npylm)
end

function sample_λ_with_initial_params(npylm::NPYLM)
    for i in 1:NUM_WORD_TYPES
        # scale = 1/rate
        dist = Gamma(npylm.λ_a, 1/npylm.λ_b)
        npylm.λ_for_types[i] = rand(dist, Float64)
    end
end

function add_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::UInt)::Bool
    @assert(n > 2)
    token_n::UInt = get_nth_word_id(sentence, n)
    pyp::PYP{UInt} = find_node_by_tracking_back_context(npylm, sentence.characters, n, npylm.whpylm_parent_p_w_cache, true, false)
    @assert pyp != nothing
    num_tables_before_addition::UInt = npylm.whpylm.root.ntables
    # TODO: This is the thing that we seem to need to keep track of. Passed in as a reference in the original code. I should probably return it in a tuple instead.
    index_of_table_added_to::UInt = 0
    word_begin_index = sentence.segment_begining_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    add_customer(pyp, token_n, npylm.whpylm_parent_p_w_cache, npylm.whpylm.d_array, npylm.whpylm.θ_array, index_of_table_added_to)
    num_tables_after_addition::UInt = npylm.whpylm.root.ntables
    # If the number of tables in the root is increased, we'll need to break down the word into characters and add them to the chpylm as well.
    # TODO: The variable namings about the depths are very confusing. Can probably do better.
    # TODO: What is this depths thing about. I still don't think I get it.
    # Yeah so I think it indeed has something to do with the CHPYLM. Will probably need to understand it a bit more by reading that original Mochihashi paper let's just see then. Eh let's see.
    # Also I think the so-called "time t" is also a bit confusing but hopefully this can somehow clear things up let's just see then.
    if (num_tables_before_addition < num_tables_after_addition)
        # Because the CHPYLM is now modified, the cache is no longer valid.
        npylm.whpylm_G_0_cache = Dict()
        # EOS is not actually a "word". Some special treatments apply.
        if token_n == EOS
            add_customer(npylm.chpylm.root, token_n, npylm.chpylm.G_0, npylm.chpylm.d_array, npylm.chpylm.θ_array, true, index_of_table_added_to)
            return true
        end
        @assert(index_of_table_added_to != 0)
        depths = npylm.prev_depth_at_table_of_token[token_n]
        # This is a new table that didn't exist before.
        @assert length(depths) < index_of_table_added_to
        prev_depths = Int[]
        add_word_to_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, npylm.most_recent_word_added_to_chpylm, prev_depths)
        @assert(length(prev_depths) == word_end_index - word_begin_index + 3)
        push!(depths, prev_depths)
    end
    return true
end

# Yeah OK so token_ids is just a temporary variable holding all the characters to be added into the chpylm? What a weird design... Why can't we do better let's see how we might refactor this code later.
function add_word_to_chpylm(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_begin_index::UInt, word_end_index::UInt, word::Vector{Char}, prev_depths::Vector{UInt})
    @assert length(prev_depths) == 0
    @assert word_end_index >= word_begin_index
    # This is probably to avoid EOS?
    @assert word_end_index <= npylm.max_sentence_length
    produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index, word)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    for n in 1:word_length_with_symbols
        depth_n = sample_depth_at_index_n(npylm.chpylm, word, npylm.chpylm.parent_p_w_cache, npylm.chpylm.path_nodes)
        add_customer_at_index_n(npylm.chpylm, word, n, depth_n, npylm.chpylm.parent_p_w_cache, npylm.path_nodes)
        push!(prev_depths, depth_n)
    end
end

function remove_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::UInt)
    @assert n > 2
    token_n = get_nth_word_id(sentence, n)
    pyp = find_node_by_tracing_back_context_from_index_n(npylm, sentence.word_ids, sentence.num_segments, n, false, false)
    @assert pyp != nothing
    num_tables_before_removal::UInt = npylm.whpylm.root.ntables
    index_of_table_removed_from = 0
    word_begin_index = sentence.segment_begining_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    remove_customer(pyp, token_n, index_of_table_removed_from)

    num_tables_after_removal::UInt = npylm.whpylm.root.ntables
    if num_tables_before_removal > num_tables_after_removal
        # The CHPYLM is changed, so we need to clear the cache.
        npylm.whpylm_G_0_cache = Dict()
        if token_n == EOS
            # EOS is not decomposable. It only gets added to the root node.
            remove_customer(npylm.chpylm.root, token_n, true, index_of_table_removed_from)
            return true
        end
        @assert index_of_table_removed_from != 0
        depths = npylm.prev_depth_at_table_of_token[token_n]
        @assert index_of_table_removed_from <= length(depths)
        prev_depths = depths[index_of_table_removed_from]
        @assert length(prev_depths > 0)
        remove_word_from_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, npylm.most_recent_word_added_to_chpylm, prev_depths)
        # This entry is now removed.
        delete!(depths, index_of_table_removed_from)
    end
    if need_to_remove_from_parent(pyp)
        remove_from_parent(pyp)
    end
    return true
end

function remove_word_from_chpylm(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_begin_index::UInt, word_end_index::UInt, word::Vector{Char}, prev_depths::Vector{UInt})
    @assert length(prev_depths) > 0
    @assert word_end_index >= word_begin_index
    @assert word_end_index <= npylm.max_sentence_length
    produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index, word)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    @assert length(prev_depths == word_length_with_symbols)
    for n in 1:word_length_with_symbols
        remove_customer_at_index_n(npylm.chpylm, word, n, prev_depths[n])
    end
end

function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, word_ids::Vector{UInt}, n::UInt, generate_if_not_found::Bool, return_middle_node::Bool)
    # TODO: These all need to change when the bigram model is supported.
    @assert n > 2
    @assert n < length(word_ids)
    cur_node = npylm.hpylm.root
    # TODO: Why only 2?
    for depth in 1:2
        # There are currently two BOS tokens.
        context::UInt = BOS
        if n - depth > 0
            context = word_ids[n - depth]
        end
        child = find_child_pyp(cur_node, context, generate_if_not_found)
        if child == nothing
            if return_middle_node
                return cur_node
            end
            return nothing
        end
        cur_node = child
    end
    @assert cur_node.depth == 2
    return cur_node
end


function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_ids::Vector{UInt}, n::UInt, word_begin_index::UInt, word_end_index::UInt, parent_p_w_cache::Vector{Float64}, generate_if_not_found::Bool, return_middle_node::Bool)
    @assert n > 2
    @assert n < length(word_ids)
    @assert word_begin_index > 0
    @assert word_end_index >= word_begin_index
    cur_node = npylm.whpylm.root
    word_n_id = word_ids[n]
    parent_p_w = compute_G_0_of_word_at_index_n(npylm, sentence_as_chars, word_begin_index, word_end_index, word_n_id)
    parent_p_w_cache[1] = parent_p_w
    for depth in 1:2
        context = BOS
        if n - depth > 0
            context = word_ids[n - depth]
        end
        p_w = compute_p_w(word_n_id, parent_p_w, npylm.whpylm.d_array, npylm.whpylm.θ_array, true)
        # TODO: Should probably be depth + 1?
        parent_p_w_cache[depth] = p_w
        child = find_child_pyp(cur_node, context, generate_if_not_found)
        if child == nothing && return_middle_node == true
            return cur_node
        end
        # So the other possibility will never be triggered?
        @assert child != nothing
        parent_p_w = p_w
        cur_node = child
    end
    @assert cur_node.depth == 2
    return cur_node
end

function compute_G_0_of_word_at_index_n(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_begin_index, word_end_index, word_n_id)
    if word_n_id == EOS
        return npylm.chpylm.G_0
    end

    word_length = word_end_index - word_begin_index + 1
    # We have a cache to prevent excessive re-calculation
    G_0 = get(npylm.whpylm_G_0_cache, word_n_id, nothing)
    # However, if the word does not exist in the cache, we'll then have to do the calculation anyways.
    if G_0 == nothing
        token_ids = npylm.most_recent_word_added_to_chpylm
        produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index, token_ids)
        # Add bow and eow
        word_length_with_symbols = word_length + 2
        p_w = compute_p_w(npylm.chpylm, token_ids, word_length_with_symbols)

        # If it's the very first iteration where there isn't any word yet, we cannot compute G_0 based on the chpylm.
        if word_length > npylm.max_word_length
            npylm.whpylm_G_0_cache[word_n_id] = p_w
            return p_w
        else
            # See section 4.3: p(k|Θ) is the probability that a word of *length* k will be generated from Θ, where Θ refers to the CHPYLM.
            p_k_given_chpylm = compute_p_k_given_chpylm(word_length)

            # Here supposedly each word type will have a different Poisson parameter. I'm not sure if that's the way things work though... Let's see.
            t = 1
            λ = npylm.λ_for_types[t]
            # Deduce the word length with the Poisson distribution
            poisson_sample = sample_poisson_k_λ(word_length, λ)
            @assert poisson_sample > 0
            # This is expression (15) of the paper, where we calculate p(c1...ck)
            # expression (5): p(c1...ck) returned from the CHPYLM is exactly the "G_0" of the WHPYLM, thus the naming.
            # G_0 is the more appropriate variable naming as this is just the way it's written in the expressions.
            G_0 = p_w / p_k_given_chpylm * poisson_sample

            # Very rarely the result will exceed 1
            if !(0 < G_0 && G_0 < 1)
                for i in word_begin_index:word_end_index
                    print(sentence_as_chars[i])
                end
                print("\n")
                println(p_w)
                println(poisson_sample)
                println(p_k_given_chpylm)
                println(G_0)
                println(word_length)
            end
            npylm.whpylm_G_0_cache[word_n_id] = G_0
            return G_0
        end
    else
        # The cache already exists. No need for duplicated computation.
        return G_0
    end
end

function sample_poisson_k_λ(k::UInt, λ::Float64)
    dist = Poisson(λ)
    return pdf(dist, k)
end

function compute_p_k_given_chpylm(npylm::NPYLM, k::UInt)
    if k > npylm.max_word_length
        return 0
    end
    return npylm.chpylm_p_k[k]
end

function sample_hyperparameters(npylm::NPYLM)
    sample_hyperparameters(npylm.whpylm)
    sample_hyperparameters(npylm.chpylm)
end

function compute_log_probability_of_sentence(npylm::NPYLM, sentence::Sentence)
    sum = 0.0
    for n in 3:sentence.num_segments
        sum += log(compute_p_w_of_nth_word(npylm, sentence, n))
    end
    return sum
end

# Do we really need two versions of this function. Apparently one would suffice?
function compute_probability_of_sentence(sentence::Sentence)
    prod = 0.0
    for n in 3:sentence.num_segments
        prod *= compute_p_w_of_nth_word(sentence, n)
    end
    return prod
end

# This is the real "compute_p_w"... The above ones don't have much to do with p_w I reckon. They are about whole sentences. Eh.
function compute_p_w_of_nth_word(npylm::NPYLM, sentence_as_chars::Vector{Char}, n::UInt)
    word_begin_index = sentence_as_chars.segment_starting_positions[n]
    # I mean, why don't you just record the end index directly anyways. The current implementation is such a torture.
    word_end_index = word_begin_index + sentence_as_chars.segment_lengths[n] - 1
    return compute_p_w_of_nth_word(npylm, sentence_as_chars.sentence_string, sentence_as_chars.word_ids, sentence_as_chars.num_segments, n, word_begin_index, word_end_index)
end

function compute_p_w_of_nth_word(npylm::NPYLM, sentence_string::String, word_ids::Vector{UInt}, n::UInt, word_begin_index::UInt, word_end_index::UInt)
    word_id = word_ids[n]
    
    # generate_if_not_found = false, return_middle_node = true
    node = find_node_by_tracing_back_context_from_index_n(npylm, sentence_string, word_ids, n, word_begin_index, word_end_index, npylm.whpylm_parent_p_w_cache, false, true)
    parent_pw = npylm.whpylm_parent_p_w_cache[node.depth]
    # The final `true` indicates that it's the with_parent_p_w variant of the function
    return compute_p_w(node, word_id, parent_pw, npylm.whpylm.d_array, npylm.whpylm.θ_array, true)
end