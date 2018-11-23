include("PYP.jl")
using OffsetVector

"""
Character Hierarchical Pitman-Yor Language Model 

In this case the HPYLM for characters is an infinite Markov model, different from that used for the words.
"""
mutable struct CHPYLM{T} <: HPYLM{T}
    #= Fields from the base HPYLM struct =#
    "Root PYP which has no context"
    root::PYP{T}
    "Depth of the whole HPYLM"
    depth::UInt
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
    max_depth::UInt
    # Used for high-speed computation
    sampling_table::Vector{Float64}
    parent_pw_cache::Vector{Float64}
    nodes::Vector{PYP{T}}

    #= Constructor =#
    function CHPYLM(G_0::Float64, max_depth::UInt, beta_stop::Float64, beta_pass::Float64)
        chpylm = new()
        @assert(G_0 > 0)
        # The point is just that the root node doesn't have any context, naturally, so this one should be a character that's never occurring?
        root = PYP{Char}(BOW)
        # I think theoretically the depth of a tree does begin from 0?
        root.depth = 0
        chpylm.beta_stop = beta_stop
        chpylm.beta_pass = beta_pass
        chpylm.depth = 0
        chpylm.G_0 = G_0
        chpylm.max_depth = max_depth
        chpylm.parent_pw_cache = zeros(Float64, max_depth + 1)
        chpylm.sampling_table = zeros(Float64, max_depth + 1)
        # The values should be uninitialized at this point.
        chpylm.nodes = Array(PYP{Char}, max_depth + 1)

        chpylm.d_array = OffsetVector{Float64}(undef, 0:0)
        chpylm.θ_array = OffsetVector{Float64}(undef, 0:0)
        chpylm.a_array = OffsetVector{Float64}(undef, 0:0)
        chpylm.b_array = OffsetVector{Float64}(undef, 0:0)
        chpylm.α_array = OffsetVector{Float64}(undef, 0:0)
        chpylm.β_array = OffsetVector{Float64}(undef, 0:0)
    end
end

"""
The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
- first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
- sample a new order (depth) according to the conditional probability
- add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path

This function adds the customer
"""
function add_customer_at_index_n(chpylm::CHPYLM, string_as_chars::Vector{Char}, n::UInt, depth::UInt)::Tuple{Bool,UInt}
    node = find_node_by_tracing_back_context(string_as_chars, n, depth, chpylm.parent_p_w_cache)
    char_n = string_as_chars[n]
    return add_customer(node, char_n, chpylm.parent_pw_cache, chpylm.d_array, chpylm.θ_array, true)
end

"""
This function adds the customer. See documentation above.

This is a version to be called from the NPYLM.

If the parent_pw_cache is already set, then update the path_nodes as well.
"""
function add_customer_at_index_n(chpylm::CHPYLM, characters::Vector{Char}, n::UInt, depth::UInt, parent_pw_cache::Vector{Float64}, path_nodes::Vector{PYP{Char}})::Tuple{Bool, UInt}
    @assert(0 <= depth && depth <= n)
    node::PYP{Char} = find_node_by_tracing_back_context(characters, n, depth, path_nodes)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[n - depth])
    end
    @assert(node.depth = depth)
    char_n::Char = characters[n]
    return add_customer(node, char_n, parent_pw_cache, chpylm.d_array, chpylm.θ_array, true)
end

"""
The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
- first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
- sample a new order (depth) according to the conditional probability
- add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path

This function removes the customer
"""
# Though why does the depth need to be passed in separately a kind of baffles me. Can't we directly fetch the depth just during this process? What does this even mean idneed we can definitely do better I guess. All thoasdfoa;sdf. Toa ;sdl taosdf; tasd.
function remove_customer_at_index_n(chpylm::CHPYLM, characters::Vector{Char}, n::UInt, depth::UInt)::Tuple{Bool,UInt}
    @assert(0 <= depth && depth <= n)
    node::PYP{Char} = find_node_by_tracing_back_context(characters, n, depth, false, false)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[n - depth])
    end
    @assert(node.depth = depth)
    char_n::Char = characters[n]
    remove_customer(node, char_n, true)

    # Check if the node needs to be removed
    if need_to_remove_from_parent(node)
        remove_from_parent(node)
    end
    return (true, 0)
end

# OK from what I see here the "n" looks more like the ending index or the span or something. Let's see further.
# Seems to me that the characters vector is really in reverse order? Otherwise it makes no sense that as we go deeper we actually decrease the index.
# TODO: Don't think I've fully understood this one. Still haven't cracked it yet keep going.
"""
This function finds the node of the suffix tree that contains the `n`th customer, whose order is `depth`.

Example:
[h,e,r, ,n,a,m,e]
n = 3
depth = 2
We retrace from "r" and get "h"

This version is used during `remove_customer`.
"""
function find_node_by_tracing_back_context(chpylm::CHPYLM, characters::Vector{Char}, n::UInt, depth::UInt, generate_if_not_found::Bool, return_cur_node_if_not_found::Bool)::Union{Nothing,PYP{Char}}
    # Impossible situation?
    if n < depth
        return nothing
    end

    # Note that we start from the root.
    cur_node = chpylm.root
    for d in 1:depth
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
    @assert(cur_node.depth == depth)
    if depth > 0
        @assert(cur_node.context == characters[n - depth])
    end
    return cur_node
end

"""
This function finds the node of the suffix tree that contains the `n`th customer, whose order is `depth`.

Example:
[h,e,r, ,n,a,m,e]
n = 3
depth = 2
We retrace from "r" and get "h"

This version is used during `add_customer`.

Cache the probabilities during the tracing.
"""
function find_node_by_tracing_back_context(chpylm::CHPYLM, characters::Vector{Char}, n::UInt, depth::UInt, parent_p_w_cache::Vector{Float64})::Union{Nothing,PYP{Char}}
    if n < depth
        return nothing
    end

end


function sample_depth_at_index_n(chpylm::CHPYLM, word::Vector{Char}, n::UInt, parent_p_w_cache::Vector{Float64}, path_nodes::Vector{PYP{Char}})
    if (n == 1)
        return 0
    end
    char_n = word[n]
    sum = 0.0
    parent_p_w = chpylm.G_0
    parent_pass_probability = 1.0
    parent_p_w_cache[1] = chpylm.G_0
    sampling_table_size = 0
    cur_node = chpylm.root
    for depth in 0:n
        if cur_node != nothing
            @assert depth == cur_node.depth
            # TODO: Finish this one after reading the infinite Markov model paper better.
        end
    end
end