include("Corpus.jl")
include("PYP.jl")

"""
Hierarchical Pitman-Yor Language Model in general

In the C++ code it's actually the base class for both CHPYLM and WHPYLM.

Here I can either go with composition just as what I did with CRP or do the duplication work. Unfortunately fields in abstract types are still not supported yet. 
See: https://github.com/JuliaLang/julia/issues/4935

Therefore two copies of the fields exist in WHPYLM and CHPYLM structs, respectivelj
"""
# I can still write out the code first. Refactoring is not that hard.
abstract type HPYLM{T} end

# TODO: This function seems to be unnecessary in Julia. Let's see.
# function delete_node(hpylm::HPYLM, pyp::PYP)
#     for child in pyp.children
#         delete_node(hpylm, child)
#     end
#     delete!(hpylm, pyp)
# end

function get_num_nodes(hpylm::HPYLM)::UInt where T
    # The root node itself is not included in this recursive algorithm which counts the number of children of a node.
    return get_num_nodes(hpylm.root) + 1
end

function get_num_tables(hpylm::HPYLM)::UInt where T
    return get_num_tables(hpylm.root)
end

function get_num_customers(hpylm::HPYLM)::UInt where T
    return get_num_customers(hpylm.root)
end

function get_pass_counts(hpylm::HPYLM)::UInt where T
    get_pass_counts(hpylm.root)
end

function get_stop_counts(hpylm::HPYLM)::UInt where T
    get_stop_counts(hpylm.root)
end

# Really?
function set_G_0(hpylm::HPYLM, G_0::Float64)
    hpylm.G_0 = G_0
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

# The `bottom` here was a reference in the C++ code.
# This one sums up all values of a auxiliary variable on the same depth into one variable.
function sum_auxiliary_variables_recursively(hpylm::HPYLM, node::PYP{T}, sum_log_x_u_array::Vector{Float64}, sum_y_ui_array::Vector{Float64}, sum_one_minus_y_ui_array::Vector{Float64}, sum_one_minus_z_uwkj_array::Vector{Float64}, bottom::UInt) where T
    for child in node.children
        depth = child.depth
        if depth > bottom
            bottom = depth
        end
        init_hyperparameters_at_depth_if_needed(hpylm, depth)

        d = hpylm.d_array[depth]
        θ = hpylm.θ_array[depth]
        sum_log_x_u_array[depth] += sample_log_x_u(node, θ)
        sum_y_ui_array[depth] += sample_summed_y_ui(node, d, θ, false)
        # true means is_one_minus
        sum_one_minus_y_ui_array[depth] += sample_summed_y_ui(node, d, θ, true)
        sum_one_minus_z_uwkj_array[depth] += sample_summed_one_minus_z_uwkj(node, d)

        # The bottom should be a locally modified reference I hope, so that the results still turn out correct. Let's see.
        sum_auxiliary_variables_recursively(child, sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array, bottom)
    end
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