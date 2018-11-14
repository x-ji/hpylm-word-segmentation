include("Corpus.jl")
include("PYP.jl")

"""
Hierarchical Pitman-Yor Language Model in general

In the C++ code it's actually the base class for both CHPYLM and WHPYLM.

Here I can either go with composition just as what I did with CRP or do the duplication work. Unfortunately fields in abstract types are still not supported yet. 
See: https://github.com/JuliaLang/julia/issues/4935
"""
# I can still write out the code first. Refactoring is not that hard.
mutable struct HPYLM{T}
    "Root PYP which has no context"
    root::PYP{T}
    "Depth of the whole HPYLM"
    depth::UInt
    "Base probability for 0-grams, i.e. G_0(w)"
    g0::Float64
    "Array of discount parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    d_array::Vector{Float64}
    "Array of concentration parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    θ_array::Vector{Float64}

    #=
    These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)

    Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
    =#
    "For the sampling of discount d"
    a_array::Vector{Float64}
    "For the sampling of discount d"
    b_array::Vector{Float64}
    "For the sampling of concentration θ"
    α_array::Vector{Float64}
    "For the sampling of concentration θ"
    β_array::Vector{Float64}

    function HPYLM(dataset::Dataset, max_word_length::UInt)
        hpylm = new()
        # Is this necessary?
        set_locale(hpylm)
        max_sentence_length = get_max_sentence_length(dataset)
    end
end

# TODO: This function seems to be unnecessary in Julia. Let's see.
# function delete_node(hpylm::HPYLM, pyp::PYP)
#     for child in pyp.children
#         delete_node(hpylm, child)
#     end
#     delete!(hpylm, pyp)
# end

function get_num_nodes(hpylm::HPYLM{T})::UInt where T
    # The root node itself is not included in this recursive algorithm which counts the number of children of a node.
    return get_num_nodes(hpylm.root) + 1
end

function get_num_tables(hpylm::HPYLM{T})::UInt where T
    return get_num_tables(hpylm.root)
end

function get_num_customers(hpylm::HPYLM{T})::UInt where T
    return get_num_customers(hpylm.root)
end

function get_pass_counts(hpylm::HPYLM{T})::UInt where T
    get_pass_counts(hpylm.root)
end

function get_stop_counts(hpylm::HPYLM{T})::UInt where T
    get_stop_counts(hpylm.root)
end

# Really?
function set_g0(hpylm::HPYLM, g0::Float64)
    hpylm.g0 = g0
end

# TODO: Again, maybe we can do without this function.
function init_hyperparameters_at_depth_if_needed(hpylm::HPYLM, depth::UInt)
    if length(hpylm.d_array) <= depth
        while(length(hpylm.d_array) <= depth)
            push!(hpylm.d_array, HPYLM_INITIAL_d)
        end
    end
    if length(hpylm.θ_array) <= depth
        while(length(hpylm.θ_array) <= depth)
            push!(hpylm.θ_array, HPYLM_INITIAL_θ)
        end
    end
    if length(hpylm.a_array) <= depth
        while(length(hpylm.a_array) <= depth)
            push!(hpylm.a_array, HPYLM_a)
        end
    end
    if length(hpylm.b_array) <= depth
        while(length(hpylm.b_array) <= depth)
            push!(hpylm.b_array, HPYLM_b)
        end
    end
    if length(hpylm.α_array) <= depth
        while(length(hpylm.α_array) <= depth)
            push!(hpylm.α_array, HPYLM_α)
        end
    end
    if length(hpylm.β_array) <= depth
        while(length(hpylm.β_array) <= depth)
            push!(hpylm.β_array, HPYLM_β)
        end
    end
end

# TODO: Finish this function
function sum_auxiliary_variables_recursively()
    
end

function sample_hyperparameters(hpylm::HPYLM)
    max_depth::UInt = length(hpylm.d_array) - 1

    # By definition depth of a tree begins at 0. Therefore we need to + 1 to get the length.
    # As shown in expression (41)
    sum_log_x_u::Vector{Float64} = zeros(Float64, max_depth + 1)
    # In expression (41)
    sum_y_ui::Vector{Float64} = zeros(Float64, max_depth + 1)
    # In expression (40)
    # I don't think we actually need this though unless we go through two sampling runs. Let's see.
    sum_one_minus_y_ui::Vector{Float64} = zeros(Float64, max_depth + 1)
    # As shown in expression (40)
    sum_one_minus_z_uwkj::Vector{Float64} = zeros(Float64, max_depth + 1)

    # First sample the values of the root.
    sum_log_x_u[1] = sample_log_x_u(hpylm.root, hpylm.θ_array[1])
    sum_y_ui[1] = sample_summed_y_ui(hpylm.root, hpylm.d_array[1], hpylm.θ_array[1], false)
    sum_one_minus_y_ui[1] = sample_summed_y_ui(hpylm.root, hpylm.d_array[1], hpylm.θ_array[1], true)
    sum_one_minus_z_uwkj[1] = sample_summed_one_minus_z_uwkj(hpylm.root, hpylm.d_array[1])
end