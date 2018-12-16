# include("PYP.jl")
using OffsetArrays
using StatsBase

"""
Character Hierarchical Pitman-Yor Language Model 

In this case the HPYLM for characters is an infinite Markov model, different from that used for the words.
"""
mutable struct CHPYLM{T} <: HPYLM{T}
    #= Fields from the base HPYLM struct =#
    "Root PYP which has no context"
    root::PYP{Char}
    "Depth of the whole HPYLM"
    depth::Int
    "Base probability for 0-grams, i.e. G_0(w)"
    G_0::Float64
    "Array of discount parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    d_array::OffsetVector{Float64}
    "Array of concentration parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    θ_array::OffsetVector{Float64}

    #=
    These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)

    Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
    =#
    "For the sampling of discount d"
    a_array::OffsetVector{Float64}
    "For the sampling of discount d"
    b_array::OffsetVector{Float64}
    "For the sampling of concentration θ"
    α_array::OffsetVector{Float64}
    "For the sampling of concentration θ"
    β_array::OffsetVector{Float64}
    #= End fields from HPYLM =#

    #= Fields specific to CHPYLM =#
    """
    q_i is the penetration probability of the node. q_i ∼ Beta(α, β)
    
    this variable represents the β
    """
    beta_stop::Float64
    """
    q_i is the penetration probability of the node. q_i ∼ Beta(α, β)
    
    this variable represents the α
    """
    beta_pass::Float64
    "This essentially corresponds to the maximum word length L specified during training"
    max_depth::Int
    # I don't think we really need a sampling table though, seeing that we can probably just run a native sampling function directly. Let's see.
    sampling_table::OffsetVector{Float64}
    # Used for high-speed computation
    parent_pw_cache::OffsetVector{Float64}
    "This is also just a temporary variable to hold the path nodes during a lookup process."
    path_nodes::OffsetVector{PYP{Char}}

    #= Constructor =#
    function CHPYLM(G_0::Float64, max_depth::Int, beta_stop::Float64, beta_pass::Float64)
        chpylm = new{Char}()
        @assert(G_0 > 0)
        # The point is just that the root node doesn't have any context, naturally, so this one should be a character that's never occurring?
        chpylm.root = PYP(BOW)
        # I think theoretically the depth of a tree does begin from 0?
        chpylm.root.depth = 0
        chpylm.beta_stop = beta_stop
        chpylm.beta_pass = beta_pass
        chpylm.depth = 0
        chpylm.G_0 = G_0
        chpylm.max_depth = max_depth
        # chpylm.parent_pw_cache = zeros(Float64, max_depth + 1)
        chpylm.parent_pw_cache = OffsetVector{Float64}(undef, 0:max_depth)
        # chpylm.sampling_table = zeros(Float64, max_depth + 1)
        chpylm.sampling_table = OffsetVector{Float64}(undef, 0:max_depth)
        chpylm.path_nodes = OffsetVector{PYP{Char}}(undef, 0:max_depth)

        chpylm.d_array = OffsetVector{Float64}(undef, 0:-1)
        chpylm.θ_array = OffsetVector{Float64}(undef, 0:-1)
        chpylm.a_array = OffsetVector{Float64}(undef, 0:-1)
        chpylm.b_array = OffsetVector{Float64}(undef, 0:-1)
        chpylm.α_array = OffsetVector{Float64}(undef, 0:-1)
        chpylm.β_array = OffsetVector{Float64}(undef, 0:-1)
        return chpylm
    end
end

"""
The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
- first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
- sample a new order (depth) according to the conditional probability
- add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path

This function adds the customer
"""
function add_customer_at_index_n(chpylm::CHPYLM, string_as_chars::OffsetVector{Char}, n::Int, depth::Int)::Bool
    node = find_node_by_tracing_back_context(string_as_chars, n, depth, chpylm.parent_p_w_cache)
    char_n = string_as_chars[n]
    root_table_index::IntContainer = IntContainer(0)
    return add_customer(node, char_n, chpylm.parent_pw_cache, chpylm.d_array, chpylm.θ_array, true, root_table_index)
end

"""
This function adds the customer. See documentation above.

This is a version to be called from the NPYLM.

If the parent_pw_cache is already set, then update the path_nodes as well.
"""
function add_customer_at_index_n(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, depth::Int, parent_pw_cache::OffsetVector{Float64}, path_nodes::OffsetVector{PYP{Char}})::Bool
    @assert(0 <= depth && depth <= n)
    node::PYP{Char} = find_node_by_tracing_back_context(chpylm, characters, n, depth, path_nodes)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[n - depth])
    end
    @assert(node.depth = depth)
    char_n::Char = characters[n]
    root_table_index::IntContainer = IntContainer(0)
    return add_customer(node, char_n, parent_pw_cache, chpylm.d_array, chpylm.θ_array, true, root_table_index)
end

"""
The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
- first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
- sample a new order (depth) according to the conditional probability
- add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path

This function removes the customer
"""
# Though why does the depth need to be passed in separately a kind of baffles me.
function remove_customer_at_index_n(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, depth::Int)::Bool
    @assert(0 <= depth && depth <= n)
    node::PYP{Char} = find_node_by_tracing_back_context(characters, n, depth, false, false)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[n - depth])
    end
    @assert(node.depth = depth)
    char_n::Char = characters[n]
    root_table_index::IntContainer = IntContainer(0)
    remove_customer(node, char_n, true, root_table_index)

    # Check if the node needs to be removed
    if need_to_remove_from_parent(node)
        remove_from_parent(node)
    end
    return true
end

"""
For the nth customer, this function finds the node with depth `depth_of_n` in the suffix tree.
The found node contains the correct contexts of length `depth_of_n` for the nth customer.

Example:
[h,e,r, ,n,a,m,e]
n = 3
depth_of_n = 2

The customer is "r". With a `depth_of_n` of 2, We should get the node for "h".
When we connect the node all the way up, we can reconstruct the full 2-gram context "h-e-" that is supposed to have generated the customer "r".

This version is used during `remove_customer`.
"""
# TODO: Try to reduce code duplication as much as possible here.
function find_node_by_tracing_back_context(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, depth_of_n::Int, generate_if_not_found::Bool, return_cur_node_if_not_found::Bool)::Union{Nothing,PYP{Char}}
    # This situation makes no sense, otherwise we'll go straight out of the start of the word.
    if n < depth_of_n
        return nothing
    end

    # Note that we start from the root.
    cur_node = chpylm.root
    for d in 1:depth_of_n
        context::Char = characters[n - d]
        # Find the child pyp whose context is the given context
        child::Union{Nothing, PYP{Char}} = find_child_pyp(cur_node, context, generate_if_not_found)
        if child == nothing
            if return_cur_node_if_not_found
                return cur_node
            else
                return nothing
            end
        else
            # Then, using that child pyp as the starting point, find its child which contains the context one further back again.
            cur_node = child
        end
    end

    # The search has ended for the whole depth.
    # In this situation the cur_node should have the same depth as the given depth.
    @assert(cur_node.depth == depth_of_n)
    if depth_of_n > 0
        @assert(cur_node.context == characters[n - depth_of_n])
    end
    return cur_node
end

"""
For the nth customer, this function finds the node with depth `depth_of_n` in the suffix tree.
The found node contains the correct contexts of length `depth_of_n` for the nth customer.

Example:
[h,e,r, ,n,a,m,e]
n = 3
depth_of_n = 2

The customer is "r". With a `depth_of_n` of 2, We should get the node for "h".
When we connect the node all the way up, we can reconstruct the full 2-gram context "h-e-" that is supposed to have generated the customer "r".

This version is used during `add_customer`. It cachees the probabilities of generating the nth customer at each level of the tree, during the tracing.
"""
# TODO: The use of parent_p_w_cache is confusing. Either keep it as a field or always operate on it as an external variable, eh?
function find_node_by_tracing_back_context(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, depth_of_n::Int, parent_p_w_cache::OffsetVector{Float64})::Union{Nothing,PYP{Char}}
    # This situation makes no sense, otherwise we'll go straight out of the start of the word.
    if n < depth_of_n
        return nothing
    end

    # The actual char at location n of the sentence
    char_n = characters[n]

    # Start from the root node, order 0
    cur_node = chpylm.root
    parent_p_w = chpylm.G_0
    parent_p_w_cache[0] = parent_p_w
    for depth in 1:depth_of_n
        # What is the possibility of char_n being generated from cur_node (i.e. having cur_node as its context).
        p_w = compute_p_w_with_parent_p_w(cur_node, char_n, parent_p_w, chpylm.d_array, chpylm.θ_array)
        parent_p_w_cache[depth] = p_w

        # The context `depth`-order before the target char
        context_char = characters[n - depth]
        # We should be able to find the PYP containing that context char as a child of the current node. If it doesn't exist yet, create it.
        child = find_child_pyp(cur_node, context_char, true)
        parent_p_w = p_w
        cur_node = child
    end
    return cur_node
end

function find_node_by_tracing_back_context(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, depth_of_n::Int, path_nodes_cache::OffsetVector{PYP{Char}})::Union{Nothing,PYP{Char}}
    # This situation makes no sense, otherwise we'll go straight out of the start of the word.
    if n < depth_of_n
        return nothing
    end
    cur_node = chpylm.root
    for depth in 1:depth_of_n
        if path_nodes_cache[depth] != nothing
            cur_node = path_nodes_cache[depth]
        else
            context_char = characters[n - depth]
            child = find_child_pyp(cur_node, context_char, true)
            cur_node = child
        end
    end
    return cur_node
end

function compute_p_w(chpylm::CHPYLM, characters::OffsetVector{Char})
    return exp(compute_log_p_w(chpylm, characters))
end

# It seems to have been mentioned that this is a relatively inefficient way to do so. Maybe we can do better?
# This one is definitely fucked without including the "actual" length of that characters array. What the actual hell is this design anyways. Eh.
function compute_log_p_w(chpylm::CHPYLM, characters::OffsetVector{Char})
    char = characters[0]
    log_p_w = 0.0
    # I still haven't fully wrapped my head around the inclusions and exclusions of BOS, EOS, BOW, EOW, etc. Let's see how this works out though.
    if char != BOW
        log_p_w += log(compute_p_w(chpylm.root, char, chpylm.G_0, chpylm.d_array, chpylm.θ_array))
    end

    for n in 1:length(characters) - 1
        # I sense that the way this calculation is written is simply not very efficient. Surely we can do better than this?
        # n - 1 because that argument is the end of the context `h`, not the actual word itself.
        # I sense another indexing error previously here. Why would it start from 0 instead of 1?
        log_p_w += log(compute_p_w_given_h(chpylm, characters, 0, n - 1))
    end

    return log_p_w
end

"Compute the probability of generating the character `characters[end + 1]` with `characters[begin:end]` as the context."
# TODO: Improve the efficiency of these calls.
function compute_p_w_given_h(chpylm::CHPYLM, characters::OffsetVector{Char}, context_begin::Int, context_end::Int)
    target_char = characters[context_end + 1]
    return compute_p_w_given_h(chpylm, target_char, characters, context_begin, context_end)
end

function compute_p_w_given_h(chpylm::CHPYLM, target_char::Char, characters::OffsetVector{Char}, context_begin::Int, context_end::Int)
    cur_node = chpylm.root
    parent_pass_probability = 1.0
    p = 0.0
    # We start from the root of the tree.
    parent_p_w = chpylm.G_0
    p_stop = 1.0
    depth = 0

    # There might be calculations that result in depths greater than the actual context length.
    while (p_stop > CHPYLM_ϵ)
        # If there is no node yet, use the Beta prior to calculate
        if cur_node == nothing
            p_stop = (chpylm.beta_stop) / (chpylm.beta_pass + chpylm.beta_stop) * parent_pass_probability
            p += parent_p_w * p_stop
            parent_pass_probability *= (chpylm.beta_pass) / (chpylm.beta_pass + chpylm.beta_stop)
        else
            p_w = compute_p_w_with_parent_p_w(cur_node, target_char, parent_p_w, chpylm.d_array, chpylm.θ_array)
            p_stop = stop_probability(cur_node, chpylm.beta_stop, chpylm.beta_pass, false) * parent_pass_probability
            p += p_w * p_stop
            parent_pass_probability *= pass_probability(cur_node, chpylm.beta_stop, chpylm.beta_pass, false)
            parent_p_w = p_w

            # Preparation for the next round.
            # We do this only in the else branch because if the cur_node is already `nothing`, it will just keep being `nothing` from then onwards.

            # If wee've already gone so deep that the depth is greater than the actual context length. Of course there's no next node at such a depth.
            # Note that this operation is with regards to the next node, thus the + 1 on the left hand side.
            # On the right hand side the + 1 is because we're getting the length, which requires + 1
            # So the two +1's are not the same!
            if depth + 1 >= context_end - context_begin + 1
                cur_node = nothing
            else
                cur_context_char = characters[context_end - depth]
                child = find_child_pyp(cur_node, cur_context_char)
                cur_node = child
            end
        end
        depth += 1
    end
    @assert p > 0.0
    return p
end

function sample_depth_at_index_n(chpylm::CHPYLM, characters::OffsetVector{Char}, n::Int, parent_p_w_cache::OffsetVector{Float64}, path_nodes::OffsetVector{PYP{Char}})
    # The first character should always be the BOW
    if (n == 1)
        return 0
    end
    char_n = characters[n]
    sum = 0.0
    parent_p_w = chpylm.G_0
    parent_pass_probability = 1.0
    parent_p_w_cache[0] = parent_p_w
    sampling_table_size = 0
    cur_node = chpylm.root
    for index in 0:n
        # Already gone beyond the original word's context length.
        if cur_node == nothing
            p_stop = (chpylm.beta_stop) / (chpylm.beta_pass + chpylm.beta_stop) * parent_pass_probability
            # If there's already no new context char, we just use the parent word probability as it is.
            p = parent_p_w * p_stop
            # The node with depth n is the parent of the node with depth n + 1. Therefore the index + 1 here.
            parent_p_w_cache[index + 1] = parent_p_w
            chpylm.sampling_table[index] = p
            path_nodes[index] = nothing
            sampling_table_size += 1
            sum += p
            parent_pass_probability *= (chpylm.beta_pass) / (chpylm.beta_pass + chpylm.beta_stop)
            if (p_stop < CHPYLM_ϵ)
                break
            end
        else
            p_w = compute_p_w_with_parent_p_w(cur_node, char_n, parent_p_w, chpylm.d_array, chpylm.θ_array)
            p_stop = stop_probability(cur_node, chpylm.beta_stop, chpylm.beta_pass, false)
            p = p_w * p_stop * parent_pass_probability
            parent_p_w = p_w
            parent_p_w_cache[index + 1] = parent_p_w
            chpylm.sampling_table[index] = p
            path_nodes[index] = cur_node
            sampling_table_size += 1
            parent_pass_probability *= pass_probability(cur_node, chpylm.beta_stop, chpylm.beta_pass, false)
            sum += p
            if (p_stop < CHPYLM_ϵ)
                break
            end
            if index < n
                context_char = characters[n - index - 1]
                cur_node = find_child_pyp(cur_node, context_char)
            end
        end
    end
    # The following samples the depth according to their respective probabilities.
    depths = [0:sampling_table_size - 1]
    return sample(depths, Weights(parent(chpylm.sampling_table)))

    # normalizer = 1.0 / sum
    # bernoulli = rand(Float64)
    # stack = 0.0
    # for i in 0:sampling_table_size - 1
    #     stack += chpylm.sampling_table[i] * normalizer
    #     if bernoulli < stack
    #         return i
    #     end
    # end
    # I think this should be sampling_table_size - 1, not the content.
    # return chpylm.sampling_table[sampling_table_size - 1]
end
