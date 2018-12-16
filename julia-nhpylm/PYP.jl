using Compat, Random, Distributions, OffsetArrays

# include("Def.jl")

# TODO: Just don't use this function and initialize d_array and θ_array to be really large from the get go.
function init_hyperparameters_at_depth_if_needed(depth::Int, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64})
    # The depth is dynamically increasing. Therefore we might need to push new hyperparameters into the array if needed.
    # However, why don't we on the contrary just initialize an array of ridiculous depth that this should never become a problem? I think that might make the operations a bit more efficient
    if depth >= length(d_array)
        while(length(d_array) <= depth)
            push!(parent(d_array), HPYLM_INITIAL_d)
        end
        while(length(θ_array) <= depth)
            push!(parent(θ_array), HPYLM_INITIAL_θ)
        end
    end
end

"""
Each node is essentially a Pitman-Yor process in the hierarchical Pitman-Yor language model

We use a type parameter because it can be either for characters (Char) or for words (UTF32String/Int?)

The root PYP (depth 0) contains zero context. The deeper the depth, the longer the context.
"""
mutable struct PYP{T}
    """
    Directly keep track of the children PYPs.

    The key in the Dict is the *additional* context to be *prepended to* the context up to now represented by this PYP.

    For example, when the current node represents the 1-gram context "will", the keys might be "he" or "she", etc., leading to nodes representing the 2-gram contexts "he will", "she will" etc.
    """
    children::Dict{T, PYP{T}}

    "Directly keep track of the parent PYP"
    parent::Union{Nothing, PYP{T}}

    """
    `tablegroups` is a `Dict` that groups the tables by the dish served. The key of the `Dict` is the dish, and the value of the `Dict` is a tablegroup, more specifically, an array which contains the customer count for each individual table in this table group.

    In this model, each table serves only one dish, i.e. the draw of that word that follows the previous context ``G_u``. However, multiple tables might serve *the same dish*, i.e. a future draw might come up with the same word as a previous draw.

    This is why we need a Vector to contain all those different tables serving this same dish (key)
    """
    tablegroups::Dict{T,Vector{Int}}

    """
    This is just a convenient variable to keep track of the total number of table groups, i.e. unique dishes, present in this `CRP` so far.

    I'm actually not very sure now whether it is for table **groups** or **tables**. Let me go on and see how the cold unfolds then.
    """
    # ntablegroups::Int

    """
    This keeps track of the number of total tables (not just table groups)
    """
    ntables::Int

    """
    +This is a convenient variable to keep track of the total number of customers for each unique dish. The key of the `Dict` is the dish, while the value of the `Dict` is the total number of customers for this dish.+

    +This is helpful because as mentioned above, there could be multiple tables serving the same dish, and we need to add those counts together.+

    In the case of the C++ implementation, there is only a total number, while to get the individual number for a particular dish one will have to do some computation.

    TODO: I can experiment to see if the other way round is faster. For now let me follow the C++ implementation.
    """
    # ncustomers::Dict{T,Int}
    ncustomers::Int

    "Useful only for CHPYLM. The number of times that the process has stopped at this Node."
    stop_count::Int

    "Useful only for CHPYLM. The number of times that the process has passed through this node."
    pass_count::Int

    """
    The depth of this PYP node in the hierarchical structure.

    Note that by definition the depth of a tree begins from 0
    """
    depth::Int

    """
    Each PYP represents a particular context.
    
    For the root node the context is ϵ.

    Only the context char/word *at this level* is stored in this struct. To construct the complete context corresponding to this PYP, we'll have to trace all the way up to the root.

    For example, a depth-2 node might store "she", while its parent, a depth-1 node, stores "will", whose parent, the root (depth-0) node, stores ϵ.

    Then, the complete context will be the 2-gram "she will".
    """
    context::T

    function PYP(context::T) where T
        pyp = new{T}()
        pyp.children = Dict{T,PYP{T}}()
        pyp.parent = nothing
        pyp.tablegroups = Dict{T, Vector{Int}}()
        # pyp.ntablegroups = 0
        pyp.ntables = 0
        pyp.ncustomers = 0
        pyp.stop_count = 0
        pyp.pass_count = 0
        # pyp.depth = 0
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
function get_num_tables_serving_dish(pyp::PYP{T}, dish::T)::Int where T
    return length(get(pyp.tablegroups, dish, []))
end

function get_num_customers_for_dish(pyp::PYP{T},dish::T)::Int where T
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
function find_child_pyp(pyp::PYP{T}, dish::T, generate_if_not_found::Bool=false)::Union{Nothing, PYP{T}} where T
    result = get(pyp.children, dish, nothing)
    if result == nothing && generate_if_not_found
        child = PYP(dish)
        child.parent = pyp
        child.depth = pyp.depth + 1
        pyp.children[dish] = child
        return child
    else
        # In this case the result is just nothing.
        return result
    end
end

# I think we won't need to duplicate the code. Just use a union type. Let's see if that works.
"The second item returned in the tuple is the index of the table to which the customer is added."
function add_customer_to_table(pyp::PYP{T}, dish::T, table_index::Int, G_0_or_parent_pws::Union{Float64, OffsetVector{Float64}}, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64}, table_index_in_root::IntContainer)::Bool where T
    tablegroup = get(pyp.tablegroups, dish, nothing)
    # println("in add_customer_to_table, tablegroup is $(tablegroup), table_index is $(table_index)")

    if tablegroup == nothing
        # println("in add_customer_to_table, tablegroup is nothing?")
        return add_customer_to_new_table(pyp, dish, G_0_or_parent_pws, d_array, θ_array, table_index_in_root);
    end

    tablegroup[table_index] += 1
    # TODO: Experiment to see which approach is faster.
    # pyp.ncustomers[dish] += 1
    pyp.ncustomers += 1
    return true
end

function add_customer_to_new_table(pyp::PYP{T}, dish::T, G_0_or_parent_pws::Union{Float64, OffsetVector{Float64}}, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64}, table_index_in_root::IntContainer)::Bool where T
    add_customer_to_new_table(pyp, dish)
    if pyp.parent != nothing
        success = add_customer(pyp, dish, G_0_or_parent_pws, d_array, θ_array, false, table_index_in_root)
        @assert(success == true)
    end
    return true;
end

function add_customer_to_new_table(pyp::PYP{T}, dish::T) where T
    tablegroup = get(pyp.tablegroups, dish, nothing)

    if tablegroup == nothing
        pyp.tablegroups[dish] = [1]
    else
        push!(tablegroup, 1)
    end

    pyp.ntables += 1
    pyp.ncustomers += 1
end

function remove_customer_from_table(pyp::PYP{T}, dish::T, table_index::Int, table_index_in_root::IntContainer) where T
    # The tablegroup should always be found.
    tablegroup = pyp.tablegroups[dish]

    # `tablegroup` is currently a Vector, so we use <=
    @assert(table_index <= length(tablegroup))
    tablegroup[table_index] -= 1
    pyp.ncustomers -= 1
    @assert(tablegroup[table_index] >= 0)
    if (tablegroup[table_index] == 0)
        if (pyp.parent != nothing)
            success = remove_customer(pyp.parent, dish, false, table_index_in_root)
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
function add_customer(pyp::PYP{T}, dish::T, G_0_or_parent_pws::Union{Float64, OffsetVector{Float64}}, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64}, update_beta_count::Bool, index_of_table_in_root::IntContainer)::Bool where T
    # println("We're in add_customer, the dish is $(dish)")
    init_hyperparameters_at_depth_if_needed(pyp.depth, d_array, θ_array)
    # Need to + 1 because by definition depth starts from 0 but array indexing starts from 1
    d_u = d_array[pyp.depth]
    θ_u = θ_array[pyp.depth]
    # println("What the hell")
    parent_pw::Float64 =
    # It seems that we do need to initialize this separately since sometimes this can result in nothing? Though why?
    if typeof(G_0_or_parent_pws) == Float64
        if pyp.parent != nothing
            compute_p_w(pyp.parent, dish, G_0_or_parent_pws, d_array, θ_array)
        else 
            G_0_or_parent_pws
        end
    # elseif typeof(G_0_or_parent_pws) == OffsetVector{Float64}
    # It must be of another type anyways.
    else
        G_0_or_parent_pws[pyp.depth]
    end

    tablegroup = get(pyp.tablegroups, dish, nothing)
    if tablegroup == nothing
        add_customer_to_new_table(pyp, dish, G_0_or_parent_pws, d_array, θ_array, index_of_table_in_root)
        if (update_beta_count)
            increment_stop_count(pyp)
        end
        # Root PYP
        if (pyp.depth == 0)
            # I think the thing is that if the table doesn't exist in root then of course this index will be given as 0.
            # TODO: Wait, should this be 0 or be 1? Should be 1 right?
            index_of_table_in_root.int = 1
        end
        return true
    else
        sum::Float64 = 0
        for k in 1:length(tablegroup)
            sum += max(0.0, tablegroup[k] - d_u)
        end
        t_u::Float64 = pyp.ntables
        sum += (θ_u + d_u * t_u) * parent_pw

        normalizer::Float64 = 1.0 / sum
        bernoulli::Float64 = rand(Float64)
        stack::Float64 = 0

        # We can add it to anywhere of the existing table
        for k in 1:length(tablegroup)
            stack += max(0.0, tablegroup[k] - d_u) * normalizer
            if bernoulli <= stack
                add_customer_to_table(pyp, dish, k, G_0_or_parent_pws, d_array, θ_array, index_of_table_in_root)
                if update_beta_count
                    increment_stop_count(pyp)
                end
                if pyp.depth == 0
                    index_of_table_in_root.int = k
                end

                return true
            end
        end

        # If we went through the whole loop but still haven't returned, we know that we should add it to a new table.

        add_customer_to_new_table(pyp, dish, G_0_or_parent_pws, d_array, θ_array, index_of_table_in_root)

        if update_beta_count
            increment_stop_count(pyp)
        end

        # In this case, we added it to the newly created table, thus set the index as such.
        if pyp.depth == 0
            # TODO: Wait, isn't this the same as the last already existing table? Is this even correct?
            index_of_table_in_root.int = length(tablegroup)
        end

        return true
    end
end

function remove_customer(pyp::PYP{T}, dish::T, update_beta_count::Bool, index_of_table_in_root::IntContainer)::Bool where T
    tablegroup = get(pyp.tablegroups, dish, nothing)
    @assert tablegroup != nothing
    println(tablegroup)
    count = sum(tablegroup)

    normalizer::Float64 = 1.0 / count
    bernoulli::Float64 = rand(Float64)
    stack = 0.0
    for k in 1:length(tablegroup)
        stack += tablegroup[k] * normalizer
        if bernoulli <= stack
            # Does it really need to keep track of the exact index of the table in root?
            remove_customer_from_table(pyp, dish, k, index_of_table_in_root)
            if update_beta_count
                decrement_stop_count(pyp)
            end
            if pyp.depth == 0
                index_of_table_in_root = k
            end
            return true
        end
    end
    # If we went through the whole tablegroup without picking one, we have to remove it from the last one anyways.
    # Basically a repeat of the above procedure. Any way to simplify this code?
    # Can definitely just use a sampling method, right?
    # TODO: Use the built-in sampling method on tablegroup[k] instead of this shit.
    remove_customer_from_table(pyp, dish, length(tablegroup), index_of_table_in_root)
    if update_beta_count
        decrement_stop_count(pyp)
    end
    if pyp.depth == 0
        index_of_table_in_root = length(tablegroup)
    end
    return true
end

function compute_p_w(pyp::PYP{T}, dish::T, G_0::Float64, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64}) where T
    println("In compute_p_w, dish is $dish")
    init_hyperparameters_at_depth_if_needed(pyp.depth, d_array, θ_array)
    d_u = d_array[pyp.depth]
    θ_u = θ_array[pyp.depth]
    t_u = pyp.ntables
    c_u = pyp.ncustomers
    println("d_u is $d_u, θ_u is $θ_u, t_u is $t_u, c_u is $c_u")
    tablegroup = get(pyp.tablegroups, dish, nothing)
    if tablegroup == nothing
        println(pyp.tablegroups)
        println("In compute_p_w, tablegroup == nothing triggered")
        coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        # If we already have parent_p_w then of course we shouldn't need to go through this again.
        if pyp.parent != nothing
            return compute_p_w(pyp.parent, dish, G_0, d_array, θ_array) * coeff
        else
            return G_0 * coeff
        end
    else
        println("In compute_p_w, tablegroup != nothing, dish is $(dish)")
        parent_p_w = G_0
        if (pyp.parent != nothing)
            parent_p_w = compute_p_w(pyp.parent, dish, G_0, d_array, θ_array)
        end
        c_uw = sum(tablegroup)
        t_uw = length(tablegroup)
        println("c_uw is $c_uw, c_uw is $t_uw")
        first_term::Float64 = max(0.0, c_uw - d_u * t_uw) / (θ_u + c_u)
        second_coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        println("first_term is $first_term, second_coeff is $second_coeff")
        return first_term + second_coeff * parent_p_w
    end
end

# Note that I added a final Bool argument to indicate whether the thing is already parent_pw or is G_0, so that I don't end up duplicating the  method.
"""
Compute the possibility of the word/char `dish` being generated from this pyp (i.e. having this pyp as its context)

When is_parent_pw == True, the third argument is the parent_p_w. Otherwise it's simply the G_0.
"""
function compute_p_w_with_parent_p_w(pyp::PYP{T}, dish::T, parent_p_w::Float64, d_array::OffsetVector{Float64}, θ_array::OffsetVector{Float64}) where T
    # println("In compute_p_w_with_parent_p_w, dish is $dish")
    init_hyperparameters_at_depth_if_needed(pyp.depth, d_array, θ_array)
    d_u = d_array[pyp.depth]
    θ_u = θ_array[pyp.depth]
    t_u = pyp.ntables
    c_u = pyp.ncustomers
    # println("d_u is $d_u, θ_u is $θ_u, t_u is $t_u, c_u is $c_u")
    tablegroup = get(pyp.tablegroups, dish, nothing)
    if tablegroup == nothing
        # println("In compute_p_w_with_parent_p_w, tablegroup == nothing triggered")
        coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        return parent_p_w * coeff
    else
        println("In compute_p_w_with_parent_p_w, tablegroup != nothing, dish is $(dish)")
        c_uw = sum(tablegroup)
        t_uw = length(tablegroup)
        println("c_uw is $c_uw, c_uw is $t_uw")
        first_term::Float64 = max(0.0, c_uw - d_u * t_uw) / (θ_u + c_u)
        second_coeff::Float64 = (θ_u + d_u * t_u) / (θ_u + c_u)
        println("first_term is $first_term, second_coeff is $second_coeff")
        return first_term + second_coeff * parent_p_w
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
function get_max_depth(pyp::PYP{T}, base::Int)::Int where T
    max_depth::Int = base
    for child in pyp.children
        depth = get_max_depth(child, base + 1)
        if (depth > max_depth)
            max_depth = depth
        end
    end
    return max_depth
end

"A DFS to get the total number of nodes with this `pyp` as the root"
function get_num_nodes(pyp::PYP{T})::Int where T
    count::Int = length(pyp.children)
    for child in pyp.children
        count += get_num_nodes(child)
    end
    # OK it seems that we can' really return + 1 here since only the root node is the special case.
    return count
end

function get_num_tables(pyp::PYP{T})::Int where T
    # The "length" of each tablegroup is exactly the "total number of tables" in that group.
    # TODO: Do without the unnecessary summation
    count = sum(length, pyp.tablegroups)
    # Do we really need this assertion then. Apparently we can just directly use that variable instead of this one?
    @assert(count == pyp.ntables)
    count += sum(get_num_tables, pyp.children)
    return count
end

function get_num_customers(pyp::PYP{T})::Int where T
    # TODO: Do without the unnecessary summation
    # The type of tablegroups is Dict{T, Vector{Int}}
    # count::Int = sum(Iterators.flatten(values(pyp.tablegroups)))
    # count::Int = sum([sum(tablegroup) for tablegroup in values(pyp.tablegroups) if length(tablegroup) > 0])
    # Whatever this should always hold let's just substitute it.
    # @assert(count== pyp.ncustomers)
    # count += sum(get_num_customers, pyp.children)
    # return count
    # return pyp.ncustomers + sum(get_num_customers, values(pyp.children))
    # return pyp.ncustomers + sum([get_num_customers(child) for child in values(pyp.children) if !isempty(child)])

    # What the hell didn't expect it to be this complicated. Fuck that let me just write an imperative loop anyways.
    temp = pyp.ncustomers
    for child in values(pyp.children)
        if !isempty(child)
            temp += child.ncustomers
        end
    end
    return temp
end

# TODO: Not using a function for this is probably faster.
function get_pass_counts(pyp::PYP{T})::Int where T
    return pyp.pass_count + sum(get_pass_counts, pyp.children)
end

function get_stop_counts(pyp::PYP{T})::Int where T
    return pyp.stop_count + sum(get_stop_counts, pyp.children)
end

"If run successfully, this function should put all pyps at the specified depth into the accumulator vector."
function get_all_pyps_at_depth(pyp::PYP{T}, depth::Int, accumulator::OffsetVector{PYP{T}}) where T
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
        return log(rand(dist))
    else
        return 0.0
    end
end

"""
Note that in expressions (40) and (41) of the technical report, the yui values are only used when they're summed. So we do the same here.
"""
function sample_summed_y_ui(pyp::PYP{T}, d_u::Float64, θ_u::Float64, is_one_minus::Bool=false)::Float64 where T
    if pyp.ntables >= 2
        sum::Float64 = 0.0
        # The upper bound is (t_u. - 1)
        for i in 1:pyp.ntables - 1
            # Only this index value i is used in the sampling, apparently.
            denom = θ_u + d_u * i
            @assert(denom > 0)
            prob = θ_u / denom
            dist = Bernoulli(prob)
            y_ui = rand(dist)
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

raw"""
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
                    sum += 1 - rand(dist)
                end
            end
        end
    end
    return sum
end
