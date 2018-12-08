# include("Def.jl")
# include("Corpus.jl")
# include("Sentence.jl")
# include("PYP.jl")
# include("HPYLM.jl")
# include("WHPYLM.jl")
# include("CHPYLM.jl")
using Distributions
using OffsetArrays

mutable struct NPYLM
    "The hierarhical Pitman-Yor model for words"
    whpylm::WHPYLM
    "The hierarhical Pitman-Yor model for characters"
    chpylm::CHPYLM

    """
    Each key represents a word.

    Remember that we feed in a word like a "sentence", i.e. run `add_customer` char-by-char, into the CHPYLM.

    We need to record the depth of each char when it was first added.

    Remember that each dish can be served at multiple tables, i.e. there is a certain probability that a customer sits at a new table.
    Therefore, the outermost Vector in Vector{Vector{Int}} keeps tracks of the different tables that this token is served at!
    Compare it with the field `tablegroups::Dict{T, Vector{Int}}` in PYP.jl

    For the *innermost* Vector{Int}, the index of the entry corresponds to the char index, the value of the entry corresponds to the depth of that particular char entry.
    This is to say, for *a particular table* that the token is served at, the breakdown of char depths is recorded in that vector.
    """
    recorded_depth_arrays_for_tablegroups_of_token::Dict{Int, Vector{Vector{Int}}}

    "The cache of WHPYLM G_0. This will be invalidated once the seating arrangements in the CHPYLM change."
    whpylm_G_0_cache::Dict{UInt, Float64}
    "The cache of CHPYLM G_0"
    chpylm_G_0_cache::Dict{Int, Float64}
    λ_for_types::Vector{Float64}
    "Probability of generating a word of length k from the CHPYLM"
    p_k_chpylm::OffsetVector{Float64}
    max_word_length::Int
    max_sentence_length::Int
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
    whpylm_parent_p_w_cache::OffsetVector{Float64}
    """
    Cache for the characters that make up the sentence that was last added to the chpylm

    Note that because this is a container that gets reused over and over again, its length is simply the maximum word length. The "actual word length" will be computed and passed in as a parameter to functions that use this variable. Not sure if this is the most sensible approach though. Maybe we can refactor it to a less tedious way later. The extra length parameter is no fun.
    """
    most_recent_word_added_to_chpylm::Vector{Char}

    function NPYLM(max_word_length::Int, max_sentence_length::Int, G_0::Float64, initial_λ_a::Float64, initial_λ_b::Float64, chpylm_beta_stop::Float64, chpylm_beta_pass::Float64)
        npylm = new()

        npylm.whpylm = WHPYLM(3)
        npylm.chpylm = CHPYLM(G_0, max_sentence_length, chpylm_beta_stop, chpylm_beta_pass)

        npylm.recorded_depth_arrays_for_tablegroups_of_token = Dict{Int, Vector{Vector{Int}}}()
        npylm.whpylm_G_0_cache = Dict{UInt, Float64}()
        npylm.chpylm_G_0_cache = Dict{Int, Float64}()

        # TODO: Expand upon word types and use different poisson distributions for different types.
        npylm.λ_for_types = zeros(Float64, NUM_WORD_TYPES)
        # Currently we use a trigram model.
        # TODO: bigram model
        npylm.whpylm_parent_p_w_cache = fill(0.0, 0:2)
        set_λ_prior(npylm, initial_λ_a, initial_λ_b)

        npylm.max_sentence_length = max_sentence_length
        npylm.max_word_length = max_word_length
        # + 2 because of bow and eow
        # Not sure if this is the most sensible approach with Julia. Surely we can adjust for that.
        npylm.most_recent_word_added_to_chpylm = Vector{Char}(undef, max_sentence_length + 2)
        # There are two extra cases where k = 1 and k > max_word_length
        # We initialize them with a uniform distribution. Later we'll use a Monte Carlo sampling to update the estimates (in function update_p_k_given_chpylm)
        # npylm.p_k_chpylm = OffsetVector{Float64}((1.0 / (max_word_length + 1)), 0:max_word_length)
        npylm.p_k_chpylm = fill(1.0 / (max_word_length + 1), 0:max_word_length)
        # for k in 0:max_word_length + 1
        #     npylm.p_k_chpylm[k] = 1.0 / (max_word_length + 1)
        # end
        return npylm
    end
end

function produce_word_with_bow_and_eow(sentence_as_chars::Vector{Char}, word_begin_index::Int, word_end_index::Int, word::Vector{Char})
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


function extend_capacity(npylm::NPYLM, max_sentence_length::Int)
    if (max_sentence_length <= npylm.max_sentence_length)
        return
    else
        allocate_capacity(npylm, max_sentence_length)
    end
end

function allocate_capacity(npylm::NPYLM, max_sentence_length::Int)
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
        npylm.λ_for_types[i] = rand(dist)
    end
end

function add_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::Int)::Bool
    @assert(n > 2)
    token_n::UInt = get_nth_word_id(sentence, n)
    pyp::PYP{UInt} = find_node_by_tracing_back_context_from_index_n(npylm, sentence, n, npylm.whpylm_parent_p_w_cache, true, false)
    @assert pyp != nothing
    num_tables_before_addition::Int = npylm.whpylm.root.ntables
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    (_, index_of_table_added_to_in_root) = add_customer(pyp, token_n, npylm.whpylm_parent_p_w_cache, npylm.whpylm.d_array, npylm.whpylm.θ_array, true)
    num_tables_after_addition::Int = npylm.whpylm.root.ntables
    # If the number of tables in the root is increased, we'll need to break down the word into characters and add them to the chpylm as well.
    # Remember that a customer has a certain probability to sit at a new table. However, it might also join an old table, in which case the G_0 doesn't change?
    if (num_tables_before_addition < num_tables_after_addition)
        # Because the CHPYLM is now modified, the cache is no longer valid.
        npylm.whpylm_G_0_cache = Dict()
        # EOS is not actually a "word". Therefore, it will always be set to be generated by the root node of the CHPYLM.
        if token_n == EOS
            add_customer(npylm.chpylm.root, token_n, npylm.chpylm.G_0, npylm.chpylm.d_array, npylm.chpylm.θ_array, true, index_of_table_added_to_in_root)
            return true
        end
        @assert(index_of_table_added_to_in_root != 0)
        # Get the depths recorded for each table in the tablegroup of token_n.
        depth_arrays_for_the_tablegroup = npylm.recorded_depth_arrays_for_tablegroups_of_token[token_n]
        # This is a new table that didn't exist before *in the tablegroup for this token*.
        @assert length(depth_arrays_for_the_tablegroup) < index_of_table_added_to_in_root
        # Variable to hold the depths of each character that was added to the CHPYLM as a part of the creation of this new table.
        recorded_depth_array = Int[]
        add_word_to_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, npylm.most_recent_word_added_to_chpylm, recorded_depth_array)
        @assert(length(recorded_depth_array) == word_end_index - word_begin_index + 3)
        # Therefore we push the result of depths for *this new table* into the array.
        push!(depth_arrays_for_the_tablegroup, recorded_depth_array)
    end
    return true
end

# Yeah OK so token_ids is just a temporary variable holding all the characters to be added into the chpylm? What a weird design... Why can't we do better let's see how we might refactor this code later.
function add_word_to_chpylm(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_begin_index::Int, word_end_index::Int, word::Vector{Char}, recorded_depths::Vector{Int})
    println("In add_word_to_chpylm")
    @assert length(recorded_depths) == 0
    @assert word_end_index >= word_begin_index
    # This is probably to avoid EOS?
    @assert word_end_index <= npylm.max_sentence_length
    produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index, word)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    for n in 1:word_length_with_symbols
        depth_n = sample_depth_at_index_n(npylm.chpylm, word, npylm.chpylm.parent_p_w_cache, npylm.chpylm.path_nodes)
        add_customer_at_index_n(npylm.chpylm, word, n, depth_n, npylm.chpylm.parent_p_w_cache, npylm.path_nodes)
        push!(recorded_depths, depth_n)
    end
end

function remove_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::Int)
    @assert n > 2
    token_n = get_nth_word_id(sentence, n)
    pyp = find_node_by_tracing_back_context_from_index_n(npylm, sentence.word_ids, sentence.num_segments, n, false, false)
    @assert pyp != nothing
    num_tables_before_removal::Int = npylm.whpylm.root.ntables
    index_of_table_removed_from = 0
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    remove_customer(pyp, token_n, index_of_table_removed_from)

    num_tables_after_removal::Int = npylm.whpylm.root.ntables
    if num_tables_before_removal > num_tables_after_removal
        # The CHPYLM is changed, so we need to clear the cache.
        npylm.whpylm_G_0_cache = Dict()
        if token_n == EOS
            # EOS is not decomposable. It only gets added to the root node.
            remove_customer(npylm.chpylm.root, token_n, true, index_of_table_removed_from)
            return true
        end
        @assert index_of_table_removed_from != 0
        depths = npylm.recorded_depth_arrays_for_tablegroups_of_token[token_n]
        @assert index_of_table_removed_from <= length(depths)
        recorded_depths = depths[index_of_table_removed_from]
        @assert length(recorded_depths > 0)
        remove_word_from_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, npylm.most_recent_word_added_to_chpylm, recorded_depths)
        # This entry is now removed.
        delete!(depths, index_of_table_removed_from)
    end
    if need_to_remove_from_parent(pyp)
        remove_from_parent(pyp)
    end
    return true
end

function remove_word_from_chpylm(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_begin_index::Int, word_end_index::Int, word::Vector{Char}, recorded_depths::Vector{Int})
    @assert length(recorded_depths) > 0
    @assert word_end_index >= word_begin_index
    @assert word_end_index <= npylm.max_sentence_length
    produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index, word)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    @assert length(recorded_depths == word_length_with_symbols)
    for n in 1:word_length_with_symbols
        remove_customer_at_index_n(npylm.chpylm, word, n, recorded_depths[n])
    end
end

function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, word_ids::Vector{UInt}, n::Int, generate_if_not_found::Bool, return_middle_node::Bool)
    # TODO: These all need to change when the bigram model is supported.
    @assert n > 2
    @assert n < length(word_ids)
    cur_node = npylm.hpylm.root
    # TODO: Why only 2?
    for depth in 1:2
        # There are currently two BOS tokens.
        context::Int = BOS
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

# Used by add_customer
function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, sentence::Sentence, n::Int, parent_p_w_cache::OffsetVector{Float64}, generate_if_not_found::Bool, return_middle_node::Bool)
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    return find_node_by_tracing_back_context_from_index_n(npylm, sentence.characters, sentence.word_ids, n, word_begin_index, word_end_index, parent_p_w_cache, generate_if_not_found, return_middle_node)
end


function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_ids::Vector{UInt}, n::Int, word_begin_index::Int, word_end_index::Int, parent_p_w_cache::OffsetVector{Float64}, generate_if_not_found::Bool, return_middle_node::Bool)
    @assert n > 2
    @assert n <= length(word_ids)
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
        p_w = compute_p_w(cur_node, word_n_id, parent_p_w, npylm.whpylm.d_array, npylm.whpylm.θ_array, true)
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
        # p_w = compute_p_w(npylm.chpylm, token_ids, word_length_with_symbols)
        p_w = compute_p_w(npylm.chpylm, token_ids)

        # If it's the very first iteration where there isn't any word yet, we cannot compute G_0 based on the chpylm.
        if word_length > npylm.max_word_length
            npylm.whpylm_G_0_cache[word_n_id] = p_w
            return p_w
        else
            # See section 4.3: p(k|Θ) is the probability that a word of *length* k will be generated from Θ, where Θ refers to the CHPYLM.
            p_k_given_chpylm = compute_p_k_given_chpylm(npylm, word_length)

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
                # Now there is a bug and this branch is triggered all the time.
                # println("Very rarely the result will exceed 1")
                # for i in word_begin_index:word_end_index
                #     print(sentence_as_chars[i])
                # end
                # print("\n")
                # println(p_w)
                # println(poisson_sample)
                # println(p_k_given_chpylm)
                # println(G_0)
                # println(word_length)
            end
            npylm.whpylm_G_0_cache[word_n_id] = G_0
            return G_0
        end
    else
        # The cache already exists. No need for duplicated computation.
        return G_0
    end
end

function sample_poisson_k_λ(k::Int, λ::Float64)
    dist = Poisson(λ)
    return pdf(dist, k)
end

function compute_p_k_given_chpylm(npylm::NPYLM, k::Int)
    if k > npylm.max_word_length
        return 0
    end
    return npylm.p_k_chpylm[k]
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
function compute_probability_of_sentence(npylm::NPYLM, sentence::Sentence)
    prod = 0.0
    for n in 3:sentence.num_segments
        prod *= compute_p_w_of_nth_word(npylm, sentence, n)
    end
    return prod
end

# This is the real "compute_p_w"... The above ones don't have much to do with p_w I reckon. They are about whole sentences. Eh.
function compute_p_w_of_nth_word(npylm::NPYLM, sentence::Sentence, n::Int)
    word_begin_index = sentence.segment_begin_positions[n]
    # I mean, why don't you just record the end index directly anyways. The current implementation is such a torture.
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    return compute_p_w_of_nth_word(npylm, sentence.characters, sentence.word_ids, sentence.num_segments, n, word_begin_index, word_end_index)
end

function compute_p_w_of_nth_word(npylm::NPYLM, sentence_as_chars::Vector{Char}, word_ids::Vector{UInt}, n::Int, word_begin_index::Int, word_end_index::Int)
    word_id = word_ids[n]
    
    # generate_if_not_found = false, return_middle_node = true
    node = find_node_by_tracing_back_context_from_index_n(npylm, sentence_as_chars, word_ids, n, word_begin_index, word_end_index, npylm.whpylm_parent_p_w_cache, false, true)
    parent_pw = npylm.whpylm_parent_p_w_cache[node.depth]
    # The final `true` indicates that it's the with_parent_p_w variant of the function
    return compute_p_w(node, word_id, parent_pw, npylm.whpylm.d_array, npylm.whpylm.θ_array, true)
end