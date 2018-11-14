using Compat, Random, Distributions

include("Def.jl")

# TODO: Just don't use this function and initialize d_array and θ_array to be really large from the get go.
function init_hyperparameters_at_depth_if_needed(depth::UInt, d_array::Vector{Float64}, θ_array::Vector{Float64})
    # The depth is dynamically increasing. Therefore we might need to push new hyperparameters into the array if needed.
    # However, why don't we on the contrary just initialize an array of ridiculous depth that this should never become a problem? I think that might make the operations a bit more efficient
    if depth >= length(d_array)
        while(length(d_array) <= depth)
            push!(d_array, HPYLM_INITIAL_d)
        end
        while(length(θ_array) <= depth)
            push!(θ_array, HPYLM_INITIAL_θ)
        end
    end
end

"""
Each node is essentially a Pitman-Yor process in the hierarchical Pitman-Yor language model

We use a type parameter because it can be either for characters (Char) or for words (String/Int?)
"""
mutable struct PYP{T}
    "Directly keep track of the children PYPs"
    children::Dict{T, PYP}

    "Directly keep track of the parent PYP"
    parent::Union{Nothing, PYP{T}}

    """
    `tablegroups` is a `Dict` that groups the tables by the dish served. The key of the `Dict` is the dish, and the value of the `Dict` is an array which contains the customer count for each individual table in this table group.

    In this model, each table serves only one dish, i.e. the draw of the word that follows the context ``G_u``. However, multiple tables might serve *the same dish*, i.e. a future draw might come up with the same word as a previous draw.

    The count should be unsigned anyways.
    """
    tablegroups::Dict{T,Vector{UInt}}

    """
    This is just a convenient variable to keep track of the total number of table groups, i.e. unique dishes, present in this `CRP` so far.

    I'm actually not very sure now whether it is for table **groups** or **tables**. Let me go on and see how the cold unfolds then.
    """
    # ntablegroups::Int

    """
    This keeps track of the number of total tables (not just table groups)
    """
    ntables::UInt

    """
    +This is a convenient variable to keep track of the total number of customers for each unique dish. The key of the `Dict` is the dish, while the value of the `Dict` is the total number of customers for this dish.+

    +This is helpful because as mentioned above, there could be multiple tables serving the same dish, and we need to add those counts together.+

    In the case of the C++ implementation, there is only a total number, while to get the individual number for a particular dish one will have to do some computation.

    TODO: I can experiment to see if the other way round is faster. For now let me follow the C++ implementation.
    """
    # ncustomers::Dict{T,Int}
    ncustomers::UInt

    "Useful only for CHPYLM. The number of times that the process has stopped at this Node."
    stopcount::UInt

    "Useful only for CHPYLM. The number of times that the process has passed through this node."
    passcount::UInt

    """
    The depth of this PYP node in the hierarchical structure.

    Note that by definition the depth of a tree begins from 0
    """
    depth::UInt

    "Each PYP corresponds to a particular context."
    context::T

    function PYP(context::T)
        pyp = new()
        pyp.children = Dict{T,PYP}()
        pyp.ntablegroups = 0
        pyp.ncustomers = 0
        pyp.stopcount = 0
        pyp.passcount = 0
        pyp.context = context
        # No need to initialize parent when it's a null value.
        return pyp
    end
end

function need_to_remove_from_parent(pyp::PYP)::Bool
    # If there's no parent then of course we can't remove it from the parent
    # I'm not sure if this is the most idiomatic way.
    if pyp.parent == nothing
        return false

    # If it has no child nor customers, then remove it.
    elseif (isempty(pyp.children) && isempty(pyp.tablegroups))
        return true

    else
        return false
    end
end

"""
This function explicitly returns the number of **tables** (i.e. not customers) serving a dish!
"""
function get_num_tables_serving_dish(pyp::PYP{T}, dish::T)::UInt where T
    return length(get(pyp.tablegroups, dish, []))
end

function get_num_customers_for_dish(pyp::PYP{T},dish::T)::UInt where T
    tablegroup = get(pyp.tablegroups, dish, nothing)

    if (tablegroup == nothing)
        return 0
    else
        return sum(tablegroup)
    end
end

"""
Find the child PYP whose context is the given dish
"""
function find_child_pyp(pyp::PYP{T}, dish::T, generate_if_not_found::Bool)::Union{Nothing, PYP{T}} where T
    result = get(pyp.children, dish, nothing)
    if result == nothing && generate_if_not_found
        child = PYP(dish)
        child.parent = pyp
        child.depth = pyp.depth + 1
        pyp[dish] = child
        return child
    else
        # In this case the result is just nothing.
        return result
    end
end

# I think we won't need to duplicate the code. Just use a union type. Let's see if that works.
function add_customer_to_table(pyp::PYP{T}, dish::T, table_index::UInt, g0_or_parent_pws::Union{Float64, Vector{Float64}}, d_array::Vector{Float64}, θ_array::Vector{Float64}, index_of_table_in_root::UInt)::Bool where T
    tablegroup = get(pyp.tablegroups, dish, nothing)

    if tablegroup == nothing
        return add_customer_to_new_table(pyp, dish, g0_or_parent_pws, d_array, θ_array, index_of_table_in_root);
    end

    # Guess it's just for debugging
    @assert(table_index < length(tablegroup))
    tablegroup[table_index] += 1
    # TODO: Experiment to see which approach is faster.
    # pyp.ncustomers[dish] += 1
    pyp.ncustomers += 1
    return true
end

function add_customer_to_new_table(pyp::PYP{T}, dish::T, g0_or_parent_pws::Union{Float64, Vector{Float64}}, d_array::Vector{Float64}, θ_array::Vector{Float64}, index_of_table_in_root::UInt)::Bool where T
    add_customer_to_new_table(pyp, dish)
    if pyp.parent != nothing
        success = add_customer(dish, g0_or_parent_pws, d_array, θ_array, false, index_of_table_in_root)
        @assert(success == true)
    end
    return true;
end

function add_customer_to_new_table(pyp::PYP{T}, dish::T) where T
    # TODO: This introduces type instability but should avoid repeated lookups? Let's see.
    tablegroup = get(pyp.tablegroups, dish, nothing)

    if tablegroup == nothing
        pyp.tablegroups[dish] = [1]
    else
        push!(tablegroup, 1)
    end

    pyp.ntables += 1
    pyp.ncustomers += 1
end

function remove_customer_from_table(pyp::PYP{T}, dish::T, table_index::UInt, index_of_table_in_root::UInt) where T
    # The tablegroup should always be found.
    tablegroup = pyp.tablegroups[dish]

    @assert(table_index < length(tablegroup))
    tablegroup[table_index] -= 1
    pyp.ncustomers -= 1
    @assert(tablegroup[table_index] >= 0)
    if (tablegroup[table_index] == 0)
        if (pyp.parent != nothing)
            success = remove_customer(dish, false, index_of_table_in_root)
            @assert(success == true)
        end

        deleteat!(tablegroup, table_index)
        pyp.ntables -= 1

        if length(tablegroup) == 0
            delete!(pyp.tablegroups, dish)
            # Will also have to delete the table from the count if we use that other system.
        end
    end
    return true
end

# Right, so d_array and θ_array are really the arrays that hold *all* hyperparameters for *all levels*
# And then we're going to get the hyperparameters for this level, i.e. d_u and \theta_u from those arrays.
# Another approach to do it, for sure.
function add_customer(pyp::PYP{T}, dish::T, g0_or_parent_pws::Union{Float64, Vector{Float64}}, d_array::Vector{Float64}, θ_array::Vector{Float64}, update_beta_count::Bool, index_of_table_in_root::UInt)::Bool where T
    init_hyperparameters_at_depth_if_needed(pyp.depth, d_array, θ_array)
    # Need to + 1 because by definition depth starts from 0 but array indexing starts from 1
    d_u = d_array[pyp.depth + 1]
    θ_u = θ_array[pyp.depth + 1]
    parent_pw::Float64 = 
    if typeof(g0_or_parent_pws == Float64) 
        if pyp.parent != nothing
            compute_p_w(pyp.parent, dish, g0_or_parent_pws, d_array, θ_array)
        else 
            g0_or_parent_pws
        end
    elseif typeof(g0_or_parent_pws == Vector{Float64})
        g0_or_parent_pws[pyp.depth]
    end

    tablegroup = get(pyp.tablegroups, dish, nothing)
    if tablegroup == nothing
        add_customer_to_new_table(dish, g0_or_parent_pws, d_array, θ_array, index_of_table_in_root)
        if (update_beta_count)
            increment_stop_count(pyp)
        end
        # Root PYP
        if (pyp.depth == 0)
            # OK I think this thing is a reference so that it can be shared between the places?
            # Still why in this case return 0 instead of k though. Let's see.
            index_of_table_in_root = 0
        end
        return true
    else
        sum::Float64 = 0
        for k in 1:length(tablegroup)
            sum += max(0.0, tablegroup[k] - d_u)
        end
        t_u::Float64 = pyp.ntablegroups
        sum += (θ_u + d_u * t_u) * parent_pw

        normalizer::Float64 = 1.0 / sum
        bernoulli::Float64 = rand(Float64)
        stack::Float64 = 0

        # We can add it to anywhere of the existing table
        for k in 1:length(tablegroup)
            stack += max(0.0, tablegroup[k] - d_u) * normalizer
            if bernoulli <= stack
                add_customer_to_table(dish, k, g0_or_parent_pws, d_array, θ_array, index_of_table_in_root)
                if update_beta_count
                    increment_stop_count(pyp)
                end
                if pyp.depth == 0
                    index_of_table_in_root = k
                end

                return true
            end
        end

        # If we went through the whole loop but still haven't returned, we know that we should add it to a new table.

        add_customer_to_new_table(pyp, dish, g0_or_parent_pws, d_array, θ_array, index_of_table_in_root)

        if update_beta_count
            increment_stop_count(pyp)
        end

        # In this case, we added it to the newly created table, thus set the index as such.
        if pyp.depth == 0
            index_of_table_in_root = length(tablegroup)
        end

        return true
    end
end

function remove_customer(pyp::PYP{T}, dish::T, update_beta_count::Bool, index_of_table_in_root::Int) where T
    tablegroup = get(pyp.tablegroups, dish, nothing)
    sum = sum(tablegroup)

    normalizer::Float64 = 1.0 / sum
    bernoulli::Float64 = rand(Float64)
    stack = 0.0
    for k in 1:length(tablegroup)
        stack += tablegroup[k] * normalizer
        if bernoulli <= stack
            # Does it really need to keep track of the exact index of the table in root?
            remove_customer_from_table(dish, k, index_of_table_in_root)
            if update_beta_count
                decrement_stop_count(pyp)
            end
            if pyp.depth == 0
                index_of_table_in_root = k
            end
            return true
        end
    end
end

# Note that I added a final Bool argument to indicate whether the thing is already parent_pw or is g0
function compute_p_w(pyp::PYP{T}, dish::T, g0_or_parent_pw::Float64, d_array::Vector{Float64}, θ_array::Vector{Float64}, is_parent_pw::Bool) where T
    init_hyperparameters_at_depth_if_needed(pyp.depth, d_array, θ_array)
    d_u = d_array[pyp.depth + 1]
    θ_u = θ_array[pyp.depth + 1]
    t_u = pyp.ntablegroups
    c_u = pyp.ncustomers
    tablegroup = get(pyp.tablegroups, dish, nothing)
    if tablegroup == nothing
        coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        if pyp.parent != nothing
            return compute_p_w(pyp.parent, dish, g0_or_parent_pw, d_array, θ_array) * coeff
        else
            return g0_or_parent_pw * coeff
        end
    else
        parent_pw = 
        if is_parent_pw
            g0_or_parent_pw
        else
            if pyp.parent != nothing
                compute_p_w(pyp.parent, dish, g0_or_parent_pw, d_array, θ_array)
            else
                g0_or_parent_pw
            end
        end
        c_uw = sum(tablegroup)
        t_uw = length(tablegroup)
        first_term::Float64 = max(0.0, c_uw - d_u * t_uw) / (θ_u + c_u)
        second_coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        return first_term + second_coeff * parent_pw
    end
end

# Methods specifically related to the character variant of PYP.
# TODO: Let's see if we can use some sort of subclassing to differentiate between character PYP and word PYP. The current situation feels a bit weird? We should be able to do better than that.
# Though yeah this should not be such a big deal. If the functions are not called on word PYP they will simply not cause any harm either. Let's see.
# I can probably focus on refactoring the code for readability later, if I want. First let me try to ensure it works.

function stop_probability(pyp::PYP{T}, beta_stop::Float64, beta_pass::Float64, recursive::Bool=true) where T
    p::Float64 = (pyp.stop_count + beta_stop) / (pyp.stop_count + pyp.pass_count + beta_stop + beta_pass)
    if !recursive
        return p
    else
        if pyp.parent !== nothing
            p *= pass_probability(pyp.parent, beta_stop, beta_pass)
        end
        return p
    end
end

function pass_probability(pyp::PYP{T}, beta_stop::Float64, beta_pass::Float64, recursive::Bool=true) where T
    p::Float64 = (pyp.pass_count + beta_pass) / (pyp.stop_count + pyp.pass_count + beta_stop + beta_pass)
    if !recursive
        return p
    else
        if pyp.parent != nothing
            p *= pass_probability(pyp.parent, beta_stop, beta_pass)
        end
        return p
    end
end

function increment_stop_count(pyp::PYP{T}) where T
    pyp.stop_count += 1
    if pyp.parent != nothing
        increment_pass_count(pyp.parent)
    end
end

function decrement_stop_count(pyp::PYP{T}) where T
    pyp.stop_count -= 1
    @assert(pyp.stop_count >= 0)
    if pyp.parent != nothing
        decrement_pass_count(pyp.parent)
    end
end

function increment_pass_count(pyp::PYP{T}) where T
    pyp.pass_count += 1
    if pyp.parent != nothing
        increment_pass_count(pyp.parent)
    end
end

function decrement_pass_count(pyp::PYP{T}) where T
    pyp.pass_count -= 1
    @assert(pyp.pass_count >= 0)
    if pyp.parent != nothing
        decrement_pass_count(pyp.parent)
    end
end

function remove_from_parent(pyp::PYP{T}) where T
    if pyp.parent == nothing
        return false
    end
    delete_child_node(pyp.parent, pyp.context)
    return true
end

function delete_child_node(pyp::PYP{T}, dish::T) where T
    child = find_child_pyp(dish)
    if child != nothing
        delete!(pyp.children, dish)
    end
    if (length(pyp.children) == 0 && length(pyp.tablegroups) == 0)
        remove_from_parent(pyp)
    end
end

"Basically a DFS to get the maximum depth of the tree with this `pyp` as its root"
function get_max_depth(pyp::PYP{T}, base::UInt)::UInt where T
    max_depth::UInt = base
    for child in pyp.children
        depth = get_max_depth(child, base + 1)
        if (depth > max_depth)
            max_depth = depth
        end
    end
    return max_depth
end

"A DFS to get the total number of nodes with this `pyp` as the root"
function get_num_nodes(pyp::PYP{T})::UInt where T
    count::UInt = length(pyp.children)
    for child in pyp.children
        count += get_num_nodes(child)
    end
    # OK it seems that we can' really return + 1 here since only the root node is the special case.
    return count
end

function get_num_tables(pyp::PYP{T})::UInt where T
    # The "length" of each tablegroup is exactly the "total number of tables" in that group.
    # TODO: Do without the unnecessary summation
    count = sum(length, pyp.tablegroups)
    # Do we really need this assertion then. Apparently we can just directly use that variable instead of this one?
    @assert(count == pyp.ntables)
    count += sum(get_num_tables, pyp.children)
    return count
end

function get_num_customers(pyp::PYP{T})::UInt where T
    # TODO: Do without the unnecessary summation
    count::UInt = sum(Iterators.flatten(pyp.tablegroups))
    @assert(count== pyp.ncustomers)
    count += sum(get_num_customers, pyp.children)
    return count
end

# TODO: Not using a function for this is probably faster.
function get_pass_counts(pyp::PYP{T})::UInt where T
    return pyp.pass_count + sum(get_pass_counts, pyp.children)
end

function get_stop_counts(pyp::PYP{T})::UInt where T
    return pyp.stop_count + sum(get_stop_counts, pyp.children)
end

"If run successfully, this function should put all pyps at the specified depth into the accumulator vector."
function get_all_pyps_at_depth(pyp::PYP{T}, depth::UInt, accumulator::Vector{PYP{T}}) where T
    if pyp.depth == depth
        push!(accumulator, pyp)
        # TODO: This implementation feels a bit inefficient. If the method already pushed the PYP at this level, then self evidently the PYP at the next level will not be a target?
        # What if I just return here. Use else.
    else
        for child in pyp.children
            get_all_pyps_at_depth(child, depth, accumulator)
        end
    end
end

#= 
The functions below are related to hyperparameter (d, θ) sampling, based on the algorithm given in the Teh Technical Report

There are 3 auxiliary variables defined, x_**u**, y_**u**i, z**u**wkj.

The following methods sample them.
=#

"""
Note that only the log of x_u is used in the final sampling, expression (41) of the Teh technical report. Therefore our function also only ever calculates the log. Should be easily refactorable though.
"""
function sample_log_x_u(pyp::PYP{T}, θ_u::Float64)::Float64 where T
    if pyp.ncustomers >= 2
        dist = Beta(θ_u + 1, pyp.ncustomers - 1)
        # TODO: The C++ code added 1e-8 to the result. Is it necessary? (Apparently this is to prevent underflow, i.e. when the sampling result is 0. Can a Beta distribution sampling ever return 0? Let me run the code as it is for now.)
        return log(rand(dist, Float64))
    else
        return 0.0
    end
end

"""
Note that in expressions (40) and (41) of the technical report, the yui values are only used when they're summed. So we do the same here.
"""
function sample_summed_y_ui(pyp::PYP{T}, d_u::Float64, θ_u::Float64, is_one_minus::Bool)::Float64 where T
    if pyp.ntables >= 2
        sum::Float64 = 0.0
        # The upper bound is (t_u. - 1)
        for i in 1:pyp.ntables - 1
            # Only this index value i is used in the sampling, apparently.
            denom = θ_u + d_u * i
            @assert(denom > 0)
            prob = θ_u / denom
            dist = Bernoulli(prob)
            y_ui = rand(dist, Float64)
            if is_one_minus
                sum += (1 - y_ui)
            else
                sum += y_ui
            end
        end
        return sum
    else
        return 0.0
    end
end

"""raw
The sum is \sum_{j=1}^{c_**u**wk - 1} (1 - z_{**u**wkj}) in expression (40) of the Teh technical report.
"""
# TODO: Might just refactor this function out. The current way it's written is convenient, but can be hard to read.
function sample_summed_one_minus_z_uwkj(pyp::PYP{T}, d_u::Float64)::Float64 where T
    sum::Float64 = 0
    for tablegroup in pyp.tablegroups
        # Each element in a `tablegroup` vector stores the customer count of a particular table
        for customercount in tablegroup
            # There's also a precondition of c_uwk >= 2
            if customercount >= 2
                # Expression (38)
                for j in 1:customercount - 1
                    @assert(j - d_u > 0)
                    prob = (j - 1) / (j - d_u)
                    dist = Bernoulli(prob)
                    sum += 1 - rand(dist, Float64)
                end
            end
        end
    end
    return sum
end
