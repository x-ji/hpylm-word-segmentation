# module PYP
using Distributions
import Base.show

"Represent the uniform distribution. Used as the fallback distribution in the case of `n = 0`."
mutable struct Uniform
    K::Int
    count::Int
    function Uniform(K::Int)
        d = new()
        d.K = K
        d.count = 0
        return d
    end
end

function increment(d::Uniform, k, initialize)
    d.count += 1
end

function decrement(d::Uniform, k)
    d.count -= 1
end

function prob(d::Uniform, k::Int)
    if k >= d.K
        return 0
    end
    return 1.0 / d.K
end

function log_likelihood(d::Uniform, full)
    return -(d.count * log(d.K))
end

function resample_hyperparameters(d::Uniform, iteration)
    return (0, 0)
end

function Base.show(io::IO, d::Uniform)
    print(io, "Uniform(K=$(d.K), count = $(d.count))")
end

"""
The `CRP` struct is the core part of the program. It represents a generic Chinese Restaurant Process, i.e. it is not bound to be used only in the context of a hierarchical Pitman-Yor language model.

Its fields keep track of the dishes served, tables, and customers of the CRP.

Its functions are all internal functions in the sense that they are supposed to be called by the wrapper struct, such as the `PYP` struct described below.
"""
mutable struct CRP

    """
    `tablegroups` is a `Dict` that groups the tables by the dish served. The key of the `Dict` is the dish, and the value of the `Dict` is an array which contains the customer count for each individual table in this table group.

    In this model, each table serves only one dish, i.e. the draw of the word that follows the context ``G_u``. However, multiple tables might serve the same dish, i.e. a future draw might come up with the same word as a previous draw.
    """
    tablegroups::Dict{Int, Array{Int, 1}}

    "This is just a convenient variable to keep track of the total number of table groups, i.e. unique dishes, present in this `CRP` so far."
    ntablegroups::Int

    "This is a convenient variable to keep track of the total number of customers for each unique dish. The key of the `Dict` is the dish, while the value of the `Dict` is the total number of customers for this dish."
    ncustomers::Dict{Int, Int}

    "This is a convenient variable to keep track of the total number of customers in this `CRP`."
    totalcustomers::Int

    function CRP()
        crp = new()
        crp.tablegroups = Dict{Int, Array{Int, 1}}()
        crp.ntablegroups = 0
        crp.ncustomers = Dict{Int, Int}()
        crp.totalcustomers = 0
        return crp
    end
end

# I can make tableIndex Nullable but that's a bit unwieldy. Since this is a dynamic language anyways,
"Seats a new customer at the ``index^th`` table serving dish `dish`. An index of `0` indicates that a new table should be created instead."
function _seat_to(crp::CRP, dish::Int, table_index::Int)
    if !(haskey(crp.tablegroups, dish))
        crp.tablegroups[dish] = []
        crp.ncustomers[dish] = 0
    end
    crp.ncustomers[dish] += 1
    crp.totalcustomers += 1

    tablegroup = crp.tablegroups[dish]

    if table_index == 0
        crp.ntablegroups += 1
        push!(tablegroup, 1)
    else
        tablegroup[table_index] += 1
    end

    return (table_index == 0)
end

"Unseats a customer at the ``index^th`` table serving dish `d`. If the table (and the tablegroup) becomes empty afterwards, this function is also responsible for cleaning up."
function _unseat_from(crp::CRP, dish::Int, table_index::Int)
    crp.ncustomers[dish] -= 1
    crp.totalcustomers -= 1
    tablegroup = crp.tablegroups[dish]
    tablegroup[table_index] -= 1
    if tablegroup[table_index] == 0 # Empty table
        deleteat!(tablegroup, table_index)
        crp.ntablegroups -= 1
        if (isempty(tablegroup)) # There is already not any table that serves this dish. Remove the entry
            delete!(crp.tablegroups, dish)
            delete!(crp.ncustomers, dish)
        end

        return true
    end

    return false
end

# """
# The `PYP` struct serves as a wrapper around a `CRP` struct, in the particular context of hierarchical Pitman-Yor language model for this project.
# 
# As indicated by the formula
# 
# ```math
# G_u \sim PY(d_{|u|}, \theta_{|u|}, G_{\pi(u)})
# ```
# 
# , for *every context* there is a Pitman-Yor process that serves as its probability distribution. This struct encapsulates such a Pitman-Yor process.
# """
mutable struct PYP
    "A reference to the `CRP` struct upon which the `PYP` is based."
    crp::CRP

    # """
    # `base` essentially represents ``G_{\pi(u)}`` in the formula, i.e. the word probability vector for the context, without the earliest word.

    # For the concrete implementation:

    # - In cases where ``n > 0``, it is represented by a `BackoffBase` struct, which contains a reference to the `PYPLM` struct of order ``n - 1``, plus a specific context of length ``n - 1``, which will be used to look up the actual `PYP` in the `models` field of the referenced `PYPLM` struct.
    # - In the case where ``n = 0``, it directly points to the `Uniform` struct.

    # Because two concrete types are both possible, no type annotation is done here. A union type might be a solution though.
    # """
    base

    "This is just a reference to the `PYPPrior` struct contained within the `PYPLM` struct of order ``n``, since as mentioned above, every `PYP` struct of the same order shares the same prior. "
    prior::PYPPrior

    # Inner constructor
    function PYP(base, prior)
        pyp = new()
        pyp.crp = CRP()
        pyp.base = base
        # The `tie` method is run to ensure that the `prior` struct of order ``n`` keeps track of this `pyp` struct and makes use of it in the sampling calculations.
        tie(prior, pyp)
        pyp.prior = prior
        return pyp
    end
end

function support(pyp::PYP)
    return keys(pyp.crp.ncustomers)
end

function d(pyp::PYP)
    return discount(pyp.prior)
end

function theta(pyp::PYP)
    return strength(pyp.prior)
end

"Sample a table which serves the particular dish. Return the index of the table within the table group. If there is no such a table group containing the dish, return 0."
function _sample_table(pyp::PYP, dish::Int)
    # `tablegroups` contains table groups, each of which serves the same dish.
    if !haskey(pyp.crp.tablegroups, dish)
        return 0
    end

    # The probability of putting this draw to a new table, as shown in Equation (11) of the technical report.
    p_new = (theta(pyp)+ d(pyp) * pyp.crp.ntablegroups) * prob(pyp.base, dish)

    # Equation (11) of the technical report, c_w - dt_w, where c_w is the total number of customers for this dish, t_w is the number of tables serving this dish.
    normalized = p_new + pyp.crp.ncustomers[dish] - d(pyp) * length(pyp.crp.tablegroups[dish])
    x = rand() * normalized

    for (index, c) in enumerate(pyp.crp.tablegroups[dish])
        # If the number of customers on that table - the discount is more than x, we return that table
        if x < c - d(pyp)
            return index
        end
        x -= c - d(pyp)
    end
    return 0
end

"""
Find the index of the table in the table group that contains the ``n^th`` customer of the particular `dish`.

Note that this function will return nothing if there's actually no customer for the dish. Therefore the caller is responsible for ensuring that the arguments are sound.
"""
function _customer_table(pyp::PYP, dish::Int, n::Int)
    tablegroup = pyp.crp.tablegroups[dish]
    # There are in total m customers for this tablegroup. We provide an index n (n <= m), and we also know the how many customers each table in this tablegroup contains (c). Therefore, all we need to do is to subtract c from n until the correct table is found. Huh.
    for (i, c) in enumerate(tablegroup)
        # This should be <= since in Julia n must start from 1
        if n <= c
            return i
        end
        n -= c
    end
end

"""
Tries to add a customer for a particular dish (after the dish has been sampled). The `initialize` parameter dictates whether the customer will sit at a new table or join an already existing table.

Note that per p.18 of the technical report, there's no need to keep track of the exact table index after this routine is finished, since we can reconstruct it in the RemoveCustomer (i.e. decrement) routine.
"""
function increment(pyp::PYP, dish::Int, initialize::Bool=false)
    i =
    if initialize
        !haskey(pyp.crp.tablegroups, dish)? 0: rand(0:length(pyp.crp.tablegroups[dish]))
    else
        _sample_table(pyp, dish)
    end

    if _seat_to(pyp.crp, dish, i)
        increment(pyp.base, dish, initialize)
    end
end

"""
Removes a random customer from the table group for `dish`. If the table is cleared, then we need to also decrement the base.

It seems that the original paper called for sampling the customer to remove with probabilities proportional to ``c_uwk``, while here it's just a totally random sampling.

The caller is responsible for ensuring that `dish` already exists in the `PYP` struct.
"""
function decrement(pyp::PYP, dish::Int)
    # We remove a random customer from a table.
    i = _customer_table(pyp, dish, rand(1:pyp.crp.ncustomers[dish]))
    # If this returns true, then the table (not necessarily the whole table group though) is cleared, and we need to also update base accordingly.
    if _unseat_from(pyp.crp, dish, i)
        decrement(pyp.base, dish)
    end
end

"""
Calculates the conditional probability that the next draw will be the particular `dish`."

This corresponds to the function `WordProb(**u**, w)` on page 988 of the Teh 2006 paper and equation (11) in the Teh 2006 technical report.
"""
function prob(pyp::PYP, dish::Int)
    # New table
    w = (theta(pyp) + d(pyp) * pyp.crp.ntablegroups) * prob(pyp.base, dish)
    # Existing tablegroups
    if haskey(pyp.crp.tablegroups, dish)
        w += pyp.crp.ncustomers[dish] - d(pyp) * length(pyp.crp.tablegroups[dish])
    end
    return w / (theta(pyp) + pyp.crp.totalcustomers)
end

function log_likelihood(pyp::PYP, full::Bool=false)
    ll = if d(pyp) == 0
            (lgamma(theta(pyp)) - lgamma(theta(pyp) + pyp.crp.totalcustomers) +
                # `tablegroups` is Dict{Int, Array{Int, 1}}
                sum(lgamma, Iterators.flatten(values(pyp.crp.tablegroups))) +
                pyp.crp.ntablegroups * log(theta(pyp))
                )
        else
            (lgamma(theta(pyp)) - lgamma(theta(pyp) + pyp.crp.totalcustomers) +
                lgamma(theta(pyp) / d(pyp) + pyp.crp.ntablegroups) -
                lgamma(theta(pyp) / d(pyp)) +
                pyp.crp.ntablegroups * (log(d(pyp)) - lgamma(1 - d(pyp))) +
                sum(map(c -> lgamma(c - d(pyp)), Iterators.flatten(values(pyp.crp.tablegroups))))
                )
        end

    # Full log likelihood means adding the log likelihood of the base and the prior as well.
    if full
        ll += log_likelihood(pyp.base, true) + log_likelihood(pyp.prior)
    end

    return ll
end

function resample_hyperparameters(pyp::PYP, iteration::Int)
    return resample_hyperparameters(pyp.prior, iteration)
end

# This is the method used to sample PYP, by interlacing decrement and increment calls.
function resample_base(pyp::PYP)
    for (dish, tablegroup) in pyp.crp.tablegroups
        for n in 1:length(tablegroup)
            decrement(pyp.base, dish)
            increment(pyp.base, dish)
        end
    end

    resample_base(pyp.base)
end

function Base.show(io::IO, pyp::PYP)
    print(io, "PYP(d=$(d(pyp)), theta=$(theta(pyp)), #customers=$(pyp.crp.totalcustomers), #tablegroups=$(pyp.crp.ntablegroups), #dishes=$(length(pyp.crp.tablegroups)), Base=$(pyp.base))")
end
# end
