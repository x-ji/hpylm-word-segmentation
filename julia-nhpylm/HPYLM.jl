# include("Corpus.jl")
# include("Sentence.jl")
# include("PYP.jl")

"""
Hierarchical Pitman-Yor Language Model in general

In the C++ code it's actually the base class for both CHPYLM and WHPYLM.

Here I can either go with composition just as what I did with CRP or do the duplication work. Unfortunately fields in abstract types are still not supported yet. 
See: https://github.com/JuliaLang/julia/issues/4935

Therefore two copies of the fields exist in WHPYLM and CHPYLM structs, respectively
"""
# I can still write out the code first. Refactoring is not that hard.
abstract type HPYLM{T} end

"Get the total number of nodes in this HPYLM."
function get_num_nodes(hpylm::HPYLM)::Int where T
    # The root node itself is not included in this recursive algorithm which counts the number of children of a node. Therefore we need to add 1 in the end.
    # @ assert
    return get_num_nodes(hpylm.root) + 1
end

function get_num_tables(hpylm::HPYLM)::Int where T
    return get_num_tables(hpylm.root)
end

function get_num_customers(hpylm::HPYLM)::Int where T
    return get_num_customers(hpylm.root)
end

function get_pass_counts(hpylm::HPYLM)::Int where T
    get_pass_counts(hpylm.root)
end

function get_stop_counts(hpylm::HPYLM)::Int where T
    get_stop_counts(hpylm.root)
end

# TODO: Again, maybe we can do without this function.
"Sometimes the hyperparameter array can be shorter than the actual depth of the HPYLM (especially for CHPYLM whose depth is dynamic. In this case initialize the hyperparameters at the deeper depth."
function init_hyperparameters_at_depth_if_needed(hpylm::HPYLM, depth::Int)
    # println("In init_hyperparameters_at_depth_if_needed. depth: $depth. Type of HPYLM: $(typeof(HPYLM))")
    if length(hpylm.d_array) <= depth
        println("Length of d_array: $(length(hpylm.d_array)), depth: $depth")
        while(length(hpylm.d_array) <= depth)
            push!(parent(hpylm.d_array), HPYLM_INITIAL_d)
        end
    end
    if length(hpylm.θ_array) <= depth
        while(length(hpylm.θ_array) <= depth)
            push!(parent(hpylm.θ_array), HPYLM_INITIAL_θ)
        end
    end
    if length(hpylm.a_array) <= depth
        while(length(hpylm.a_array) <= depth)
            push!(parent(hpylm.a_array), HPYLM_a)
        end
    end
    if length(hpylm.b_array) <= depth
        while(length(hpylm.b_array) <= depth)
            push!(parent(hpylm.b_array), HPYLM_b)
        end
    end
    if length(hpylm.α_array) <= depth
        while(length(hpylm.α_array) <= depth)
            push!(parent(hpylm.α_array), HPYLM_α)
        end
    end
    if length(hpylm.β_array) <= depth
        while(length(hpylm.β_array) <= depth)
            push!(parent(hpylm.β_array), HPYLM_β)
        end
    end
end

# The `bottom` here was a reference in the C++ code.
"Sum up all values of a auxiliary variable on the same depth into one variable."
function sum_auxiliary_variables_recursively(hpylm::HPYLM, node::PYP{T}, sum_log_x_u_array::OffsetVector{Float64}, sum_y_ui_array::OffsetVector{Float64}, sum_one_minus_y_ui_array::OffsetVector{Float64}, sum_one_minus_z_uwkj_array::OffsetVector{Float64}, bottom::Int)::Int where T
    # println("Number of node.children: $(length(node.children))")
    for (context, child) in node.children
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
        bottom = sum_auxiliary_variables_recursively(hpylm, child, sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array, bottom)
    end
    return bottom
end

"Sample all hyperparameters of this HPYLM"
function sample_hyperparameters(hpylm::HPYLM)
    # The length also includes the slot for depth 0, thus the max_depth is the length - 1
    max_depth::Int = length(hpylm.d_array) - 1

    # By definition depth of a tree begins at 0. Use OffsetArray
    # As shown in expression (41)
    sum_log_x_u_array::OffsetVector{Float64} = zeros(Float64, 0:max_depth)
    # In expression (41)
    sum_y_ui_array::OffsetVector{Float64} = zeros(Float64, 0:max_depth)
    # In expression (40)
    # I don't think we actually need this though unless we go through two sampling runs. Let's see.
    sum_one_minus_y_ui_array::OffsetVector{Float64} = zeros(Float64, 0:max_depth)
    # As shown in expression (40)
    sum_one_minus_z_uwkj_array::OffsetVector{Float64} = zeros(Float64, 0:max_depth)

    # First sample the values of the root.
    sum_log_x_u_array[0] = sample_log_x_u(hpylm.root, hpylm.θ_array[0])
    sum_y_ui_array[0] = sample_summed_y_ui(hpylm.root, hpylm.d_array[0], hpylm.θ_array[0], false)
    sum_one_minus_y_ui_array[0] = sample_summed_y_ui(hpylm.root, hpylm.d_array[0], hpylm.θ_array[0], true)
    sum_one_minus_z_uwkj_array[0] = sample_summed_one_minus_z_uwkj(hpylm.root, hpylm.d_array[0])

    # I don't think you should change the depth variable of the struct itself?
    hpylm.depth = 0
    hpylm.depth = sum_auxiliary_variables_recursively(hpylm, hpylm.root, sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array, hpylm.depth)
    init_hyperparameters_at_depth_if_needed(hpylm, hpylm.depth)

    for u in 0:hpylm.depth
        # println("The current depth is $(u), the a_array value is $(hpylm.a_array[u]), the sum_one_minus_y_ui_array value is $(sum_one_minus_y_ui_array[u]), the b_array value is $(hpylm.b_array[u]), the sum_one_minus_z_uwkj_array value is $(sum_one_minus_z_uwkj_array[u])")
        dist1 = Beta(hpylm.a_array[u] + sum_one_minus_y_ui_array[u], hpylm.b_array[u] + sum_one_minus_z_uwkj_array[u])
        hpylm.d_array[u] = rand(dist1)
        
        dist2 = Gamma(hpylm.α_array[u] + sum_y_ui_array[u], 1 / (hpylm.β_array[u] - sum_log_x_u_array[u]))
        hpylm.θ_array[u] = rand(dist2)
    end

    # Delete hyperparameters at excessive depths. 
    # There could be discrepancies between the max_depth and the actual depth? I think this would only apply to CHPYLM since for WHPYLM the dpeth should always be fixed instead of being variable.
    excessive_length = max_depth - hpylm.depth
    for i in 1:excessive_length
        # They are offset arrays, so we need to use `parent` here.
        pop!(parent(hpylm.d_array))
        pop!(parent(hpylm.θ_array))
        pop!(parent(hpylm.a_array))
        pop!(parent(hpylm.b_array))
        pop!(parent(hpylm.α_array))
        pop!(parent(hpylm.β_array))
    end
end