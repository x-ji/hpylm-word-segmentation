include("PYP.jl")

"""
Character Hierarchical Pitman-Yor Language Model 

In this case the HPYLM for characters is an infinite-gram model, different from that used for the words.
"""
mutable struct CHPYLM
    #= Fields from the base HPYLM struct =#
    root::PYP
    G_0::Float64
    d_array::Vector{Float64}
    θ_array::Vector{Float64}
    depth::UInt

    #= Fields specific to CHPYLM =#
    beta_stop::Float64
    beta_pass::Float64
    "This essentially corresponds to the maximum word length L specified during training"
    max_depth::UInt
    # Used for high-speed computation
    sampling_table::Vector{Float64}
    parent_pw_cache::Vector{Float64}
    nodes::Vector{PYP}

    function CHPYLM(G_0::Float64, max_depth::UInt, beta_stop::Float64, beta_pass::Float64)
        chpylm = new()
        @assert(G_0 > 0)
        # See if this representation of the empty char works.
        root = PYP{Char}('')
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
    end
end

"To be called from the WHPYLM"
# TODO: I don't know what the "t" means yet.
function add_customer_at_t(chpylm::CHPYLM, characters::Vector{Char}, t::UInt, depth::UInt, parent_pw_cache::Vector{Float64}, nodes::Vector{PYP{Char}})
    @assert(0 <= depth && depth <= t)
    node::PYP{Char} = find_node_by_tracking_back_context(characters, t, depth, nodes)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[t - depth])
    end
    @assert(node.depth = depth)
    char_t::Char = characters[t]
    # I'm still not totally sure of the use of this one. Definitely will have to refactor the code since I don't think primitive types can be passed as a reference?
    # OK I think the whole purpose of this one variable is that it may later be used in this struct, not in the PYP struct. All the handling and passing around in PYP-related functions really are only meant so that it can be reused later in this one then. Let's see of course. Let's see.
    # TODO: This is not a correct implementation. The sensible choice would be to create a wrapper struct which contains an UInt, so that the UInt can be modified. Still I'd wait to figure out just what exactly they're trying to use this thing for...
    table_index_in_root::UInt = 0
    return add_customer(char_t, parent_pw_cache, chpylm.d_array, chpylm.θ_array, true, table_index_in_root)
end

function remove_customer_at_t(chpylm::CHPYLM, characters::Vector{Char}, t::UInt, depth::UInt)
    @assert(0 <= depth && depth <= t)
    node::PYP{Char} = find_node_by_tracking_back_context(characters, t, depth, false, false)
    # Seems to be just a check
    if depth > 0
        @assert(node.context == characters[t - depth])
    end
    @assert(node.depth = depth)
    char_t::Char = characters[t]
    # TODO: The same as above
    table_index_in_root::UInt = 0
    return remove_customer(char_t, true, table_index_in_root)

    # Check if the node needs to be removed
    if need_to_remove_from_parent(node)
        remove_from_parent(node)
    end
end

# OK from what I see here the "t" looks more like the ending index or the span or something. Let's see further.
function find_node_by_tracking_back_context(chpylm::CHPYLM, characters::Vector{Char}, t::UInt, depth::UInt, generate_if_not_found::Bool, return_cur_node_if_not_found::Bool)::Union{Nothing,PYP{Char}}
    # Impossible situation?
    if t < depth
        return nothing
    end

    cur_node = chpylm.root
    for d in 1:depth
        context::Char = characters[t - d]
        # Find the child pyp whose context is the given context
        child::Union{Nothing, PYP{Char}} = find_child_pyp(cur_node, context, generate_if_not_found)
        if child == nothing
            if return_cur_node_if_not_found
                return cur_node
            else
                return nothing
            end
        else
            cur_node = child
        end
    end

    # The search has ended for the whole depth.
    # In this situation the cur_node should have the same depth as the given depth.
    @assert(cur_node.depth == depth)
    if depth > 0
        @assert(cur_node.context == characters[t - depth])
    end
    return cur_node
end

function sample_depth_at_index_t(chpylm::CHPYLM, word::Vector{Char}, n::UInt, parent_p_w_cache::Vector{Float64}, path_nodes::Vector{PYP{Char}})
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