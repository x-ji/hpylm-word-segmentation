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

    We need to record the depth of each char when it was first added, so that we can remove them at the same depths later, when we remove the word from the CHPYLM.

    Remember that each dish can be served at multiple tables, i.e. there is a certain probability that a customer sits at a new table.
    Therefore, the outermost Vector in Vector{Vector{Int}} keeps tracks of the different tables that this token is served at!
    This is useful so that when we later remove
    Compare it with the field `tablegroups::Dict{T, Vector{Int}}` in PYP.jl

    For the *innermost* Vector{Int}, the index of the entry corresponds to the char index, the value of the entry corresponds to the depth of that particular char entry.
    This is to say, for *a particular table* that the token is served at, the breakdown of char depths is recorded in that vector.
    Normal `Vector` is used so that the first char of the word is accessed via index 1. Probably not a good idea? I guess it still works as long as the whole system is consistent. Let's see.
    Yeah I think it might be a better idea to keep these things always constant anyways. Let's see.
    """
    recorded_depth_arrays_for_tablegroups_of_token::Dict{UInt, Vector{OffsetVector{Int}}}

    "The cache of WHPYLM G_0. This will be invalidated once the seating arrangements in the CHPYLM change."
    whpylm_G_0_cache::Dict{UInt, Float64}
    "The cache of CHPYLM G_0"
    chpylm_G_0_cache::Dict{Int, Float64}
    λ_for_types::OffsetVector{Float64}
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
    most_recent_word::OffsetVector{Char}

    function NPYLM(max_word_length::Int, max_sentence_length::Int, G_0::Float64, initial_λ_a::Float64, initial_λ_b::Float64, chpylm_beta_stop::Float64, chpylm_beta_pass::Float64)
        npylm = new()

        # TODO: Also produce a bigram version.
        npylm.whpylm = WHPYLM(3)
        npylm.chpylm = CHPYLM(G_0, max_sentence_length, chpylm_beta_stop, chpylm_beta_pass)

        npylm.recorded_depth_arrays_for_tablegroups_of_token = Dict{Int, Vector{OffsetVector{Int}}}()
        npylm.whpylm_G_0_cache = Dict{UInt, Float64}()
        npylm.chpylm_G_0_cache = Dict{Int, Float64}()

        # TODO: Expand upon word types and use different poisson distributions for different types.
        npylm.λ_for_types = zeros(Float64, 0:WORDTYPE_NUM_TYPES)
        # Currently we use a trigram model.
        # TODO: bigram model
        npylm.whpylm_parent_p_w_cache = fill(0.0, 0:2)
        set_λ_prior(npylm, initial_λ_a, initial_λ_b)

        npylm.max_sentence_length = max_sentence_length
        npylm.max_word_length = max_word_length
        # + 2 because of bow and eow
        # Not sure if this is the most sensible approach with Julia. Surely we can adjust for that.
        # Yeah don't think we should always initialize it to the max length. Recipe for disaster in a sense that's for sure eh?
        # npylm.most_recent_word = OffsetVector{Char}(undef, 0:max_sentence_length + 1)
        # npylm.most_recent_word = OffsetVector{Char}(undef, 0:max_sentence_length + 1)
        # There are two extra cases where k = 1 and k > max_word_length
        # We initialize them with a uniform distribution. Later we'll use a Monte Carlo sampling to update the estimates (in function update_p_k_given_chpylm)
        # npylm.p_k_chpylm = OffsetVector{Float64}((1.0 / (max_word_length + 1)), 0:max_word_length)
        npylm.p_k_chpylm = fill(1.0 / (max_word_length + 2), 0:max_word_length + 1)
        return npylm
    end
end

"""
Wraps a character array which represents a word with two special tokens: BOW and EOW

I think a better idea is to just return an array each time which is never overly long.
"""
# function produce_word_with_bow_and_eow(sentence_as_chars::OffsetVector{Char}, word_begin_index::Int, word_end_index::Int, word::OffsetVector{Char})
#     word[0] = BOW
#     # # The length is end - begin + 1. This is always the case.
#     i = 0
#     while i < (word_end_index - word_begin_index + 1)
#         # - 1 because Julia arrays are 1-indexed
#         word[i + 1] = sentence_as_chars[word_begin_index + i]
#         i += 1
#     end
#     word[i + 1] = EOW
#     # word[2:2 + word_end_index - word_begin_index] = sentence_as_chars[word_begin_index:word_end_index]
#     # # One past the end of the word.
#     # word[word_end_index - word_begin_index + 1 + 1] = EOW
# end

function produce_word_with_bow_and_eow(sentence_as_chars::OffsetVector{Char}, word_begin_index::Int, word_end_index::Int)
    # + 2 to accomodate BOW and EOW
    word = OffsetVector{Char}(undef, 0:word_end_index - word_begin_index + 2)
    # push!(parent(word), BOW)
    # # # The length is end - begin + 1. This is always the case.
    # i = 0
    # while i < (word_end_index - word_begin_index + 1)
    #     # - 1 because Julia arrays are 1-indexed
    #     push!(parent(word), sentence_as_chars[word_begin_index + i])
    #     i += 1
    # end
    # push!(parent(word), EOW)
    # # word[2:2 + word_end_index - word_begin_index] = sentence_as_chars[word_begin_index:word_end_index]
    # # # One past the end of the word.
    # # word[word_end_index - word_begin_index + 1 + 1] = EOW
    # return word

    word[0] = BOW
    # # The length is end - begin + 1. This is always the case.
    i = 0
    while i < (word_end_index - word_begin_index + 1)
        # - 1 because Julia arrays are 1-indexed
        word[i + 1] = sentence_as_chars[word_begin_index + i]
        i += 1
    end
    word[i + 1] = EOW
    return word
end

function extend_capacity(npylm::NPYLM, max_sentence_length::Int)
    if (max_sentence_length <= npylm.max_sentence_length)
        return
    else
        allocate_capacity(npylm, max_sentence_length)
        # This line seems to be extraneous
        npylm.max_sentence_length = max_sentence_length
    end
end

function allocate_capacity(npylm::NPYLM, max_sentence_length::Int)
    npylm.max_sentence_length = max_sentence_length
    npylm.most_recent_word = OffsetVector{Char}(undef, 0:max_sentence_length + 1)
end

function set_λ_prior(npylm::NPYLM, a::Float64, b::Float64)
    npylm.λ_a = a
    npylm.λ_b = b
    sample_λ_with_initial_params(npylm)
end

function sample_λ_with_initial_params(npylm::NPYLM)
    for i in 1:WORDTYPE_NUM_TYPES
        # scale = 1/rate
        dist = Gamma(npylm.λ_a, 1/npylm.λ_b)
        npylm.λ_for_types[i] = rand(dist)
    end
end

"""
This function adds the nth segmented word in the sentence to the NPYLM.
"""
function add_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::Int)::Bool
    # The first two entries are always the BOS symbols.
    @assert(n >= 2)
    token_n::UInt = get_nth_word_id(sentence, n)
    pyp::PYP{UInt} = find_node_by_tracing_back_context_from_index_n(npylm, sentence, n, npylm.whpylm_parent_p_w_cache, true, false)
    @assert pyp != nothing
    num_tables_before_addition::Int = npylm.whpylm.root.ntables
    index_of_table_added_to_in_root::IntContainer = IntContainer(-1)
    add_customer(pyp, token_n, npylm.whpylm_parent_p_w_cache, npylm.whpylm.d_array, npylm.whpylm.θ_array, true, index_of_table_added_to_in_root)
    num_tables_after_addition::Int = npylm.whpylm.root.ntables
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    # If the number of tables in the root is increased, we'll need to break down the word into characters and add them to the chpylm as well.
    # Remember that a customer has a certain probability to sit at a new table. However, it might also join an old table, in which case the G_0 doesn't change?
    if (num_tables_before_addition < num_tables_after_addition)
        # Because the CHPYLM is now modified, the cache is no longer valid.
        npylm.whpylm_G_0_cache = Dict()
        # EOS is not actually a "word". Therefore, it will always be set to be generated by the root node of the CHPYLM.
        if token_n == EOS
            # Will need some sort of special treatment for EOS
            
            add_customer(npylm.chpylm.root, EOS_CHAR, npylm.chpylm.G_0, npylm.chpylm.d_array, npylm.chpylm.θ_array, true, index_of_table_added_to_in_root)
            return true
        end
        @assert(index_of_table_added_to_in_root.int != -1)
        # Get the depths recorded for each table in the tablegroup of token_n.
        # It may not exist yet so we'll have to check and create it if that's the case.
        depth_arrays_for_the_tablegroup = get!(npylm.recorded_depth_arrays_for_tablegroups_of_token, token_n) do
            []
        end
        # This is a new table that didn't exist before *in the tablegroup for this token*.
        @assert length(depth_arrays_for_the_tablegroup) <= index_of_table_added_to_in_root.int
        # Variable to hold the depths of each character that was added to the CHPYLM as a part of the creation of this new table.
        # recorded_depth_array = Int[]
        # `word_end_index - word_begin_index + 2` is `length(word) - 1 + 2`, i.e. word_length_with_symbols
        recorded_depth_array = OffsetVector{Int}(undef, 0:word_end_index - word_begin_index + 2)
        add_word_to_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, recorded_depth_array)
        @assert(length(recorded_depth_array) == word_end_index - word_begin_index + 3)
        # Therefore we push the result of depths for *this new table* into the array.
        push!(depth_arrays_for_the_tablegroup, recorded_depth_array)
    end
    return true
end

# Yeah OK so token_ids is just a temporary variable holding all the characters to be added into the chpylm? What a weird design... Why can't we do better let's see how we might refactor this code later.
function add_word_to_chpylm(npylm::NPYLM, sentence_as_chars::OffsetVector{Char}, word_begin_index::Int, word_end_index::Int, recorded_depths::OffsetVector{Int})
    # println("In add_word_to_chpylm")
    # @assert length(recorded_depths) == 0
    @assert word_end_index >= word_begin_index
    # This is probably to avoid EOS?
    @assert word_end_index < npylm.max_sentence_length
    npylm.most_recent_word = produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    for n in 0:word_length_with_symbols - 1
        depth_n = sample_depth_at_index_n(npylm.chpylm, npylm.most_recent_word, n, npylm.chpylm.parent_p_w_cache, npylm.chpylm.path_nodes)
        # println("depth_n sampled is $depth_n")
        add_customer_at_index_n(npylm.chpylm, npylm.most_recent_word, n, depth_n, npylm.chpylm.parent_p_w_cache, npylm.chpylm.path_nodes)
        # push!(recorded_depths, depth_n)
        recorded_depths[n] = depth_n
    end
end

function remove_customer_at_index_n(npylm::NPYLM, sentence::Sentence, n::Int)
    @assert n >= 2
    token_n = get_nth_word_id(sentence, n)
    pyp = find_node_by_tracing_back_context_from_index_n(npylm, sentence.word_ids, n, false, false)
    @assert pyp != nothing
    num_tables_before_removal::Int = npylm.whpylm.root.ntables
    index_of_table_removed_from = IntContainer(-1)
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1

    # println("In remove_customer_at_index_n, before remove_customer. token_n: $token_n, word_begin_index: $word_begin_index, word_end_index: $word_end_index")
    remove_customer(pyp, token_n, true, index_of_table_removed_from)

    num_tables_after_removal::Int = npylm.whpylm.root.ntables
    if num_tables_before_removal > num_tables_after_removal
        # The CHPYLM is changed, so we need to clear the cache.
        npylm.whpylm_G_0_cache = Dict()
        if token_n == EOS
            # EOS is not decomposable. It only gets added to the root node of the CHPYLM.
            # The char representation for EOS is what, "1"?
            remove_customer(npylm.chpylm.root, EOS_CHAR, true, index_of_table_removed_from)
            return true
        end
        @assert index_of_table_removed_from.int != -1
        depths = npylm.recorded_depth_arrays_for_tablegroups_of_token[token_n]
        recorded_depths = depths[index_of_table_removed_from.int]
        @assert length(recorded_depths) > 0
        remove_word_from_chpylm(npylm, sentence.characters, word_begin_index, word_end_index, recorded_depths)
        # This entry is now removed.
        deleteat!(depths, index_of_table_removed_from.int)
    end
    if need_to_remove_from_parent(pyp)
        remove_from_parent(pyp)
    end
    return true
end

function remove_word_from_chpylm(npylm::NPYLM, sentence_as_chars::OffsetVector{Char}, word_begin_index::Int, word_end_index::Int, recorded_depths::OffsetVector{Int})
    @assert length(recorded_depths) > 0
    @assert word_end_index >= word_begin_index
    @assert word_end_index < npylm.max_sentence_length
    npylm.most_recent_word = produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index)
    # + 2 because of bow and eow
    word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
    @assert length(recorded_depths) == word_length_with_symbols
    for n in 0:word_length_with_symbols - 1
        remove_customer_at_index_n(npylm.chpylm, npylm.most_recent_word, n, recorded_depths[n])
    end
end

function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, word_ids::OffsetVector{UInt}, n::Int, generate_if_not_found::Bool, return_middle_node::Bool)
    # TODO: These all need to change when the bigram model is supported.
    @assert n >= 2
    @assert n < length(word_ids)
    cur_node = npylm.whpylm.root
    # TODO: Why only 2?
    for depth in 1:2
        # There are currently two BOS tokens.
        context = BOS
        if n - depth >= 0
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
    @assert n >= 2
    # println("Sentence is $(sentence), n is $(n), sentence.num_segments is $(sentence.num_segments)")
    @assert n < sentence.num_segments
    @assert sentence.segment_lengths[n] > 0
    word_begin_index = sentence.segment_begin_positions[n]
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    return find_node_by_tracing_back_context_from_index_n(npylm, sentence.characters, sentence.word_ids, n, word_begin_index, word_end_index, parent_p_w_cache, generate_if_not_found, return_middle_node)
end

# We should be filling the parent_p_w_cache while trying to find the node already. So if not there's some problem going on.
function find_node_by_tracing_back_context_from_index_n(npylm::NPYLM, sentence_as_chars::OffsetVector{Char}, word_ids::OffsetVector{UInt}, n::Int, word_begin_index::Int, word_end_index::Int, parent_p_w_cache::OffsetVector{Float64}, generate_if_not_found::Bool, return_middle_node::Bool)
    @assert n >= 2
    @assert n < length(word_ids)
    @assert word_begin_index >= 0
    @assert word_end_index >= word_begin_index
    cur_node = npylm.whpylm.root
    word_n_id = word_ids[n]
    parent_p_w = compute_G_0_of_word_at_index_n(npylm, sentence_as_chars, word_begin_index, word_end_index, word_n_id)
    # println("The first parent_p_w is $parent_p_w")
    parent_p_w_cache[0] = parent_p_w
    for depth in 1:2
        context = BOS
        if n - depth >= 0
            context = word_ids[n - depth]
        end
        # println("Trying to compute_p_w_with_parent_p_w, but what is word_n_id first??? $word_n_id")
        p_w = compute_p_w_with_parent_p_w(cur_node, word_n_id, parent_p_w, npylm.whpylm.d_array, npylm.whpylm.θ_array)
        # println("The depth is $depth, the p_w is $p_w")
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

function compute_G_0_of_word_at_index_n(npylm::NPYLM, sentence_as_chars::OffsetVector{Char}, word_begin_index, word_end_index, word_n_id)
    # println("In compute_G_0_of_word_at_index_n, sentence_as_chars is $sentence_as_chars, word_begin_index is $word_begin_index, word_end_index is $word_end_index, word_n_id is $word_n_id")
    if word_n_id == EOS
        # println("The word is EOS, directly return")
        return npylm.chpylm.G_0
    end

    @assert word_end_index < npylm.max_sentence_length
    @assert word_begin_index >= 0
    @assert word_end_index >= word_begin_index
    word_length = word_end_index - word_begin_index + 1
    # We have a cache to prevent excessive re-calculation
    G_0 = get(npylm.whpylm_G_0_cache, word_n_id, nothing)
    # However, if the word does not exist in the cache, we'll then have to do the calculation anyways.
    if G_0 == nothing
        # println("The nothing branch is entered.")
        # token_ids = npylm.most_recent_word
        npylm.most_recent_word = produce_word_with_bow_and_eow(sentence_as_chars, word_begin_index, word_end_index)
        # Add bow and eow
        word_length_with_symbols = word_length + 2
        # p_w = compute_p_w(npylm.chpylm, token_ids, word_length_with_symbols)
        p_w = compute_p_w(npylm.chpylm, npylm.most_recent_word)
        # println("most_recent_word is $npylm.most_recent_word, p_w is $p_w")

        # If it's the very first iteration where there isn't any word yet, we cannot compute G_0 based on the chpylm.
        if word_length > npylm.max_word_length
            npylm.whpylm_G_0_cache[word_n_id] = p_w
            return p_w
        else
            # See section 4.3: p(k|Θ) is the probability that a word of *length* k will be generated from Θ, where Θ refers to the CHPYLM.
            p_k_given_chpylm = compute_p_k_given_chpylm(npylm, word_length)

            # Each word type will have a different poisson parameter
            t = detect_word_type(sentence_as_chars, word_begin_index, word_end_index)
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
                println("Very rarely the result will exceed 1")
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
        # println("G_0 already exists in cache, it is $G_0")
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

"""
Compute the probability of the sentence by using the product of the probabilities of the words that make up the sentence.
"""
function compute_probability_of_sentence(npylm::NPYLM, sentence::Sentence)
    prod = 1.0
    for n in 2:sentence.num_segments - 1
        prod *= compute_p_w_of_nth_word(npylm, sentence, n)
    end
    return prod
end

"""
Compute the probability of the sentence by using the sum of the log probabilities of the words that make up the sentence.

Using log could be more beneficial in preventing underflow.
"""
function compute_log_probability_of_sentence(npylm::NPYLM, sentence::Sentence)
    sum = 0.0
    for n in 2:sentence.num_segments - 1
        sum += log(compute_p_w_of_nth_word(npylm, sentence, n))
    end
    return sum
end

# This is the real "compute_p_w"... The above ones don't have much to do with p_w I reckon. They are about whole sentences. Eh.
function compute_p_w_of_nth_word(npylm::NPYLM, sentence::Sentence, n::Int)
    @assert n >= 2
    @assert n < sentence.num_segments
    @assert sentence.segment_lengths[n] > 0
    word_begin_index = sentence.segment_begin_positions[n]
    # I mean, why don't you just record the end index directly anyways. The current implementation is such a torture.
    word_end_index = word_begin_index + sentence.segment_lengths[n] - 1
    return compute_p_w_of_nth_word(npylm, sentence.characters, sentence.word_ids, n, word_begin_index, word_end_index)
end

function compute_p_w_of_nth_word(npylm::NPYLM, sentence_as_chars::OffsetVector{Char}, word_ids::OffsetVector{UInt}, n::Int, word_begin_position::Int, word_end_position::Int)
    word_id = word_ids[n]
    # println("We're in compute_p_w_of_nth_word, sentence_as_chars: $sentence_as_chars, word_ids: $word_ids, word_begin_index: $word_begin_position, word_end_index: $word_end_position")

    # So apparently the parent_p_w_cache should be set while we're trying to find the node?
    # generate_if_not_found = false, return_middle_node = true
    node = find_node_by_tracing_back_context_from_index_n(npylm, sentence_as_chars, word_ids, n, word_begin_position, word_end_position, npylm.whpylm_parent_p_w_cache, false, true)
    @assert node != nothing
    # println("Node is $node")
    parent_p_w = npylm.whpylm_parent_p_w_cache[node.depth]
    # println("The parent_p_w is $parent_p_w")
    # The final `true` indicates that it's the with_parent_p_w variant of the function
    # println("What happened?")
    return compute_p_w_with_parent_p_w(node, word_id, parent_p_w, npylm.whpylm.d_array, npylm.whpylm.θ_array)
end