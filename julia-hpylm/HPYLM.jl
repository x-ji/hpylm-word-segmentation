__precompile__()
module HPYLM
using Distributions
using SpecialFunctions
using Serialization
using Random
using StatsBase

include("Prior.jl")
include("Corpus.jl")
include("UniformDist.jl")
# include("PYP.jl")

import Base.show

# This is the vocabulary of individual characters.
# Essentially it makes no difference whether we already separate the characters in the input file beforehand, or we do it when reading in the file. Let me just assume plain Chinese text input and do it when reading in the file then.
# It's now just easier to make them global variables as they are used everywhere.
# Since there aren't really multithread operations. It should be fine.
char_vocab = Vocabulary()
word_vocab = Vocabulary()


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
    tablegroups::Dict{Int,Array{Int,1}}

    "This is just a convenient variable to keep track of the total number of table groups, i.e. unique dishes, present in this `CRP` so far."
    ntablegroups::Int

    "This is a convenient variable to keep track of the total number of customers for each unique dish. The key of the `Dict` is the dish, while the value of the `Dict` is the total number of customers for this dish."
    ncustomers::Dict{Int,Int}

    "This is a convenient variable to keep track of the total number of customers in this `CRP`."
    totalcustomers::Int

    function CRP()
        crp = new()
        crp.tablegroups = Dict{Int,Array{Int,1}}()
        crp.ntablegroups = 0
        crp.ncustomers = Dict{Int,Int}()
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

 raw"""
 The `PYP` struct serves as a wrapper around a `CRP` struct, in the particular context of hierarchical Pitman-Yor language model for this project.

 As indicated by the formula

 ```math
 G_u \sim PY(d_{|u|}, \theta_{|u|}, G_{\pi(u)})
 ```

 , for *every context* there is a Pitman-Yor process that serves as its probability distribution. This struct encapsulates such a Pitman-Yor process.
 """
mutable struct PYP
    "A reference to the `CRP` struct upon which the `PYP` is based."
    crp::CRP

    raw"""
     `base` essentially represents ``G_{\pi(u)}`` in the formula, i.e. the word probability vector for the context, without the earliest word.

     For the concrete implementation:

     - In cases where ``n > 0``, it is represented by a `BackoffBase` struct, which contains a reference to the `PYPContainer` struct of order ``n - 1``, plus a specific context of length ``n - 1``, which will be used to look up the actual `PYP` in the `models` field of the referenced `PYPContainer` struct.
     - In the case where ``n = 0``, it directly points to the `UniformDist` struct.

     Because two concrete types are both possible, no type annotation is done here. A union type might be a solution though.
     """
    base

    "This is just a reference to the `PYPPrior` struct contained within the `PYPContainer` struct of order ``n``, since as mentioned above, every `PYP` struct of the same order shares the same prior. "
    prior::PYPPrior

    "Indicates whether it is a word PYP or a char PYP. Another way to do so might be to use a union type, but let's see this one first."
    # is_for_words::Bool

    """
    Constructs a PYP struct.

    Note that it's possible for base to be a BackoffBase as well.
    """
    # function PYP(base, prior, is_for_words::Bool)
    function PYP(base, prior)
        pyp = new()
        pyp.crp = CRP()
        # I don't really know why we need a base. Can't we just directly refer to the actual PYP that resides in the field of the referenced PYPContainer? What's the potential problem with that?
        pyp.base = base
        # The `tie` method is run to ensure that the `prior` struct of order ``n`` keeps track of this `pyp` struct and makes use of it in the sampling calculations.
        tie(prior, pyp)
        pyp.prior = prior
        # pyp.is_for_words = is_for_words
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
    p_new = (theta(pyp) + d(pyp) * pyp.crp.ntablegroups) * prob(pyp.base, dish)

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
function increment(pyp::PYP, dish::Int, initialize::Bool = false)
    table_index =
    if initialize
        # An index of 0 indicates that a new table should be created instead.
        !haskey(pyp.crp.tablegroups, dish) ? 0 : rand(0:length(pyp.crp.tablegroups[dish]))
    else
        _sample_table(pyp, dish)
    end

    if _seat_to(pyp.crp, dish, table_index)
        # Send proxy customers to the base as well.
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
    # Note: In this case when pyp.base is the Uniform distribution, the result of the `prob` call here will also be that from such a distribution
    # However, if the base is a character HPYLM, how do we estimate the prob?
    # One way is to faithfully implement the infinite-gram model mentioned in the original paper
    # An approximation would be to simply multiply together the 3-gram probabilities of the whole sentence, right?
    # Maybe I should first start with this implementation anyways, and see where we get from there.
    w = (theta(pyp) + d(pyp) * pyp.crp.ntablegroups) * prob(pyp.base, dish)
    # Existing tablegroups
    if haskey(pyp.crp.tablegroups, dish)
        w += pyp.crp.ncustomers[dish] - d(pyp) * length(pyp.crp.tablegroups[dish])
    end
    return w / (theta(pyp) + pyp.crp.totalcustomers)
end

"""
Special in the sense that this is exclusively for the prob of the PYP of G_1 in the word HPYLM, i.e. the transition between word HPYLM and char HPYLM.

The special processing is that we know that pyp.base is *directly* the top-level char-type PYPContainer. There's no annoying BackoffBase struct standing in between. So we need to invoke the prob(PYPContainer, ctx, dish) method. Thus, we need to break the word dish down into a character dish.
"""
function special_prob(pyp::PYP, dish::Int)
    # How do we actually make those two things globally accessible. Huh.
    # One way would be to let each HPYLM struct hold a reference to them.
    # However this will likely create tons of overhead when performing the serialization.
    # Guess I have no other way but to put out some sort of module-level constant? Let me see if that works then.
    char_seq = string_to_charseq(dish)
    # char_dish = char_seq[end]
    # char_ctx = char_seq[1:end-1]
    # TODO: Still there's a problem in that since we haven't started using an infinite HPYLM yet, and the ctx length can well exceed (or indeed stay below) the actual length of the rigid HPYLM we just assigned there... We would need to do some sort of padding.
    # Let me just implement the actual infinite-gram model outright I guess. That would make things so much clearer. We can then also directly implement another `prob` method based on type matching, instead of having to mess around with all this nonsense.
    # OK that turned out to be a bit involved to implement. Guess I'm left without any choice but to just break down the characters into 3-grams and simply multiply the 3-gram probabilities together then. Let's see.
    char_hpylm_prob = 1.0
    char_ngrams = ngrams(char_seq, pyp.base.order)
    for ngram in char_ngrams
        # println("We're inside of special_prob, line 280")
        # println(pyp.base)
        char_hpylm_prob *= prob(pyp.base, ngram[1:end-1], ngram[end])
    end
    w = (theta(pyp) + d(pyp) * pyp.crp.ntablegroups) * char_hpylm_prob
    # Existing tablegroups
    if haskey(pyp.crp.tablegroups, dish)
        w += pyp.crp.ncustomers[dish] - d(pyp) * length(pyp.crp.tablegroups[dish])
    end
    return w / (theta(pyp) + pyp.crp.totalcustomers)
end

function log_likelihood(pyp::PYP, full::Bool = false)
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
# However this method never seems to be called for some reason.
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

raw"""
 As stated in Yee 2006 (p.987, formula (3)), given a context u, the probability vector for any word following the context u, ``G_{u}``, is
 
 ```math
 G_{u} \sim PY(d_{|u|}, \theta_{|u|}, G_{\pi(u)})
 ```
 
 where ``\pi(u)`` is the suffix of u consisting of all but the earliest word.
 
 Furthermore, ``d_{|u|}`` is the discount and ``\theta_{|u|}`` is the strength; they are both functions of the length ``|u|`` of the context.
 
 Therefore, there are ``2n`` parameters in the model. The *discount* and *strength* parameters, ``d_{|u|}`` and ``\theta_{|u|}``, are shared across all Pitman-Yor processes which have the same ``n`` value.
 
 `PYPContainer` serves as the struct that encapsulates several structs which are united *around the same `n` value*.
 """
mutable struct PYPContainer
    "a `PYPPrior` struct, which contains the *discount* and *strength* parameters."
    prior::PYPPrior

    "The order ``n``"
    order::Int

    # It's probably bad design to have two totally different types stand in for one field. Let's see if there's something that can be changed about the design.
    raw"""
    A reference to the `struct` that should be used as the basis of the `backoff` for any `PYP` contained in `models` (i.e. the `struct` on which the ``G_{\pi(u)}`` from the formula above is based).
    - In cases where ``n > 0``, it will be another `PYPContainer` struct with an `order` of ``n-1``.
    - In the case where ``n = 0``, it will be
        +the `UniformDist` struct, which corresponds to the 'global mean vector' mentioned in the paper (p. 988).+
        either:
        - A nested (character) PYPContainer (The case for word PYPContainer)
        - A UniformDist prior over all possible characters (The case for character PYPContainer)

    Because two concrete types are both possible, no type annotation is done here. A union type might be a solution though.
    """
    backoff

    "a list of `PYP` structs that have the same ```n`` value but different contexts."
    models::Dict{Array{Int,1},PYP}

    "Indicates whether it is a word PYPContainer or a char PYPContainer. Another way to do so might be to use a union type, but let's see this one first."
    is_for_words::Bool

    # These two are only useful for serializing the trained model later. NOT something that really technically belongs to the struct!
    # The serialization could be organized better, e.g. serializing a list/dict that contains both of those. This can be done later.
    # Now I'm just trying to put both of them as global variables.
    # """
    # The character vocabulary
    # """
    # char_vocab::Vocabulary

    # """
    # The word vocabulary
    # """
    # word_vocab::Vocabulary

    function PYPContainer(order::Int, initial_base, is_for_words::Bool = true)
        p = new()
        p.prior = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0) # discount = 0.8, theta = 1
        p.order = order
        # Note that if this PYPContainer is a word PYPContainer, the initial base will be a char PYPContainer, and we'll need some special processing on it.
        # I think this mostly matters when we are using pyp.base. Let's see.
        p.backoff = order == 1 ? initial_base : PYPContainer(order - 1, initial_base, is_for_words)
        p.models = Dict{Array{Int,1},PYP}()
        p.is_for_words = is_for_words
        return p
    end
end

"This is one possible type for the `base` field of the `PYP` struct. It contains a reference to the `PYPContainer` struct of order ``n - 1``, plus a specific context of length ``n - 1``, which will be used to look up the actual `PYP` in the `models` field of the referenced `PYPContainer` struct."
struct BackoffBase
    # This naming makes much more sense. This is the PYPContainer for the backoff of the PYP that contains this BackoffBase.
    pyp_container::PYPContainer
    ctx::Array{Int,1}
end

"""
Look up a `PYP` struct contained within the `PYPContainer` struct, using the context `ctx`.

If there is no such a `PYP` struct with `ctx` context, a new one will be initialized.

However, note that the `models` field will not be altered in this method! This is because `get` is not supposed to actually cause any changes. Any newly initialized `PYP` struct will only get added to `models` in `increment` method.
"""
function get(pyp_container::PYPContainer, ctx::Array{Int,1})
    if !haskey(pyp_container.models, ctx)
        # Do I really need a BackoffBase construct?... Anyways let me try to replicate vpyp structure first then.
        # The reason why we need a special BackoffBase struct here is that the `get` method doesn't actually add things to `models`. It's a kind of lazy initialization in that we provide a clue about how to calculate the base a bit later on.
        # OK I think what we need to do is to differentiate between the situations where we're dealing with a word model or a character model
        # if pyp_container.order == 1 && pyp_container.is_for_words
            # +TODO+: We need to specially construct the context, by dismantling the current word, if I get it correctly.
            # On a second thought, I don't think we actually need to modify this method. Just modifying `increment`, `decrement`, and `prob` should be enough.
            # Where the fuck is the current word though??? Seems that this function has a ton of restrictions then.
            # Doesn't the PYP struct itself contain the current word? Let me check.

            # Note that the paper actually used an infinite gram model, which seems to comprise of a series of multiplications of the stuff.
            # We need to construct a special backoffbase which has the ctx of the current word broken down, right?
            # However the way this thing is currently programmed clearly doesn't support this. What the hell.
            # OK I think I get it. The place to change is still indeed the `prob` function. However, we'd need to find a special way. Either we implement a special `prob` function, or we record more info in the PYP itself.
            # backoffbase = BackoffBase(pyp_container.backoff, )
        # else
            backoffbase = pyp_container.order == 1 ? pyp_container.backoff : BackoffBase(pyp_container.backoff, ctx[2:end])
        # end
        # The whole PYPContainer with the same order shares one prior.
        return PYP(backoffbase, pyp_container.prior)
    end
    return pyp_container.models[ctx]
end

"""
Helper function to increment

Convert a string (represented in Int) to its sequence of chars (represented in Int)
"""
# function string_to_charseq(string::Int, char_vocab::Vocabulary, word_vocab::Vocabulary)::Array{Int,1}
function string_to_charseq(str::Int)::Array{Int,1}
    global char_vocab
    global word_vocab
    # First: Convert the string from int to its original form
    word::String = get(word_vocab, str)
    # Then: Look up the characters that constitute the word one by one
    # Seems that somehow here the String is regarded as an AbstractString, and I'll need to preemptively convert the String to an Array with the `collect` method. Eh.
    return map(char -> get(char_vocab, string(char)), collect(word))
end

# Remember that a dish is a word, here the last word in the ngram.
"""
Run `increment` method on the `PYP` struct with context `ctx`, using dish `dish`.

If there is no such a `PYP` struct with `ctx` context, a new one will be initialized via `get` method, and then set in `models` field.
"""
function increment(pyp_container::PYPContainer, ctx::Array{Int,1}, dish::Int)
    # Yes I know what happens:
    # When the order of the pyp_container is 1 and the pyp_container is a word container, we should run the equivalent of `add_sentence_to_model` method, but on the decomposed representation of the `dish` as individual characters instead!
    # In this case there shouldn't be any ctx right? 
    # This is weird. Let me see what happens if the order is 1. Does this method ever get called?
    # Yeah just checked and in that case the ctx is simply `Int64[]`, which is fine.
    # The `models` field contains all PYP of order `n`. The PYPs also contain a BackoffBase of order `n - 1`.

    if pyp_container.is_for_words && pyp_container.order == 1
        char_seq = string_to_charseq(dish)
        add_sentence_to_model(pyp_container.backoff, char_seq)
    else
        if !haskey(pyp_container.models, ctx)
            pyp_container.models[ctx] = get(pyp_container, ctx)
        end
        increment(pyp_container.models[ctx], dish)
    end
end

"""
Run `decrement` method on the `PYP` struct with context `ctx`, using dish `dish`.

The caller is responsible for ensuring that the `PYP` struct with context `ctx` already exists, and that `dish` already exists in that `PYP`.
"""
function decrement(pyp_container::PYPContainer, ctx::Array{Int,1}, dish::Int)
    if pyp_container.is_for_words && pyp_container.order == 1
        char_seq = string_to_charseq(dish)
        remove_sentence_from_model(pyp_container.backoff, char_seq)
    else
        decrement(pyp_container.models[ctx], dish)
    end
end

"""
Run `prob` method on the `PYP` struct with context `ctx`, using dish `dish`, i.e. calculates the conditional probability that the next draw from that `PYP` will be this particular `dish`.

Note that unlike `decrement`, this method **doesn't require neither the `PYP` nor the `dish` to already exist!** The model is capable of calculating the probability for a previously unseen context/dish.
"""
function prob(pyp_container::PYPContainer, ctx::Array{Int,1}, dish::Int)
    # What we know:
    # The G_0(w) is either the probability derived from a properly implemented infinite-gram char HPYLM, or as a substitution, just the multiplied probability out of a 3-gram char HPYLM.
    # So OK it seems that we need to properly take care of pyp.base and perform some sort of transformation on `dish` as well then.
    # Damn I really hate the logical organization of the original vpyp code. Why did he write it this way? I'm lost all the time. This is not fun at all.
    # Though refactoring also takes time. Let me first try to ensure that this structure works, sure.
    if pyp_container.is_for_words && pyp_container.order == 1
        # In this case, the `get` method returns the PYP of G_1, which doesn't have any context to speak of whatsoever. This is to say, at G_1 level there's only *one single PYP object* contained in the `pyp_container.models` field.
        # We need a special prob method. Maybe there are better ways to do this but let me go through with this implementation at first.
        # OK let's see if this thing works!
        return special_prob(get(pyp_container, ctx), dish)
    else
        return prob(get(pyp_container, ctx), dish)
    end
end

"Recursively return the log likelihood of the whole model, including those of all the contained PYPs, those of the priors, and those from the backoff structs."
function log_likelihood(pyp_container::PYPContainer, full::Bool = false)
    return (sum(log_likelihood, values(pyp_container.models)) +
            log_likelihood(pyp_container.prior) +
            log_likelihood(pyp_container.backoff, true))
end

"""
Resample all the parameters.

It first resamples the the parameters in both the priors for this number of `n`, and then it recursively runs this method on all its backoff structs until the backoff model for `n = 0` is reached.
"""
function resample_hyperparameters(pyp_container::PYPContainer, iteration::Int)
    println("Resampling level $(pyp_container.order) hyperparameters")
    (a1, r1) = resample(pyp_container.prior, iteration)
    println("resample complete")
    (a2, r2) = resample_hyperparameters(pyp_container.backoff, iteration)
    return (a1 + a2, r1 + r2)
end

function Base.show(io::IO, pyp_container::PYPContainer)
    print(io, "PYPContainer(order=$(pyp_container.order), isForWords=$(pyp_container.is_for_words), #ctx=$(length(pyp_container.models)), prior=$(pyp_container.prior), backoff=$(pyp_container.backoff))")
end

# These are the functions related operations on the `BackoffBase` type
function increment(bb::BackoffBase, dish::Int, initialize::Bool = false)
    increment(bb.pyp_container, bb.ctx, dish)
end

function decrement(bb::BackoffBase, dish::Int)
    decrement(bb.pyp_container, bb.ctx, dish)
end

function prob(bb::BackoffBase, dish::Int)
    return prob(bb.pyp_container, bb.ctx, dish)
end

export train;
"""
Train the model on training corpus.

Arguments:
        corpus
            help=training corpus
            required=true
        order
            help=order of the model
            arg_type = Int
            required=true
        iter
            help=number of iterations for the model
            arg_type = Int
            required = true
        output
            help=model output path
            required = true
"""
function train(corpus_path, order, iter, output_path)
    println("Reading training corpus")

    # We first construct the character corpus one character at a time.
    f = open(corpus_path)
    training_corpus = read_corpus(f, char_vocab)
    close(f)

    # +I think we should actually need a Poisson correction over this?+
    # See p.102 Section 3
    # Previously, the lexicon is finite, so we could just use a uniform prior. But here, because now the lexicon are all generated from the word segmentation, the lexicon becomes *countably infinite*.
    # Here is where we need to make modifications. The initial base, i.e. G_0, in the original Teh model is just a uniform distribution. But here we need to make it another distribution, a distribution which is based on a character-level HPYLM.
    # All distributions should have the same interface, have the same set of methods that enable sampling and all that. Therefore there must be some sort of "final form" of the character HPYLM, from which this word-level HPYLM can fall back upon.
    # We just need to define and initialize that distribution somewhere.

    # The final base measure for the character HPYLM, as described in p. 102 to the right of the page.
    # This should actually be "uniform over the possible characters" of the given language. IMO this seems to suggest importing a full character set for Chinese or something. But just basing it on the training material shouldn't hurt? Let's see then.
    character_base = UniformDist(length(char_vocab))
    # False means this is for chars, not words.
    character_model = PYPContainer(3, character_base, false)
    
    # TODO: Create a special type for character n-gram model and use that directly.
    # TODO: They used Poisson distribution to correct for word length (later).
    # This is the whole npylm. Its outmost layer is the word HPYLM, while the base of the word HPYLM is the char HPYLM.
    npylm = PYPContainer(2, character_model, true)

    # initial_base = UniformDist(length(char_vocab))
    # model = PYPContainer(order, initial_base)

    println("Training model")

    # Then it's pretty much the problem of running the sampler.
    blocked_gibbs_sampler(npylm, training_corpus, iter, 100)

    # Also useful when serializing
    # npylm.char_vocab = char_vocab
    # npylm.word_vocab = word_vocab
    out = open(output_path, "w")
    # TODO: Need to serialize the vocabulary structs differently.
    serialize(out, npylm)
    close(out)
end




"""
The blocked Gibbs sampler. Function that arranges the entire sampling process for the training.

The previous approach is to just use a Gibbs sampler to randomly select a character and draw a simple binary decision about whether there's a word boundary there. Each such decision would trigger an update of the language model.
  - That was slow and would not converge without annealing. Required 20000 sampling for each character in the training data.
  - Only works on a bigram level.

Here we use a blocked Gibbs sampler:
1. A sentence is *randomly selected* (out of all the sentences in the training data).
2. Remove the unit ("sentence") data of its word segmentation from the NPYLM (it should be cascaded between the two PYLMs anyways).
  Though I suppose the "removal" can only happen starting from the second iteration, as usual.
3. Sample a new segmentation on this sentence using our sampling algorithm.
4. Add the sampled unit ("sentence") data back to the NPYLM based on this new segmentation.
"""
function blocked_gibbs_sampler(npylm::PYPContainer, corpus::Array{Array{Int,1},1}, n_iter::Int, mh_iter::Int)
    # We need to store the segmented sentences somewhere, mostly because we need to remove the sentence data when the iteration count is >= 2.
    # I might need to change the type in which the sentences are stored. for now it's Array{Array{Int,1},1}, just as in the original vpyp program: Each word is represented as an Integer, and when we need to get its representation character-by-character, we need to go through some sort of conversion process.
    # Actually a two-dimensional array might be more sensible. But I can worry about optimizations later. The corpus is read in as a nested array because of the way list comprehension works.
    total_n_sentences = length(corpus)
    # What's the way to initialize an array of objects? This should work already by allocating the space, I guess.
    segmented_sentences::Array{Array{Int,1},1} = fill(Int[], total_n_sentences)

    # This doesn't seem to be necessary
    # for i in 1:length(corpus)
    #     segmented_sentences[i] = []
    # end

    # Run the blocked Gibbs sampler for this many iterations.
    for it in 1:n_iter
        println("Iteration $it/$n_iter")

        # I think a more reasonable way is to simply shuffle an array of all the sentences, and make sure that in each iteration, every sentence is selected, albeit in a random order, of course.
        sentence_indices = shuffle(1:length(corpus))

        for index in sentence_indices
            # First remove the segmented sentence data from the NPYLM
            if it > 1
                # How do we actually do it though. Since we can't just uniformly generate fixed n-grams from this bag of characters nonchalently, it seems that we actually need to keep track of all the prevoius segmentations of the strings, if I understood it correctly.
                # Let me proceed with other parts of this function first and return to this a bit later, I guess.
                # OK. Done it. Great!
                remove_sentence_from_model(npylm, segmented_sentences[index])
            end
            # Get the raw, unsegmented sentence and segment it again.
            selected_sentence = corpus[index]
            segmented_sentence = sample_segmentation(selected_sentence, 5, npylm)
            # Add the segmented sentence data to the NPYLM
            add_sentence_to_model(npylm, segmented_sentence)
            # Store the (freshly segmented) sentence so that we may remove its segmentation data in the next iteration.
            segmented_sentences[index] = segmented_sentence
        end

        # In the paper (Figure 3) they seem to be sampling the hyperparameters at every iteration.
        println("Resampling hyperparameters")
        acceptance, rejection = resample_hyperparameters(npylm, mh_iter)
        acceptancerate = acceptance / (acceptance + rejection)
        println("MH acceptance rate: $acceptancerate")
        # println("Model: $model")
        # ll = log_likelihood(npylm)
        # perplexity = exp(-ll / (n_words + n_sentences))
        # println("ll=$ll, ppl=$perplexity")
    end

end

# These two are helper functions to the whole blocked Gibbs sampler.
function add_sentence_to_model(npylm::PYPContainer, sentence::Array{Int,1})
    sentence_ngrams = ngrams(sentence, npylm.order)
    for ngram in sentence_ngrams
        increment(npylm, ngram[1:end - 1], ngram[end])
    end
end

function remove_sentence_from_model(npylm::PYPContainer, sentence::Array{Int,1})
    sentence_ngrams = ngrams(sentence, npylm.order)
    for ngram in sentence_ngrams
        decrement(npylm, ngram[1:end - 1], ngram[end])
    end
end

"""
Function to the run the forward-backward inference which samples a sentence segmentation.

Sample a segmentation **w** for each string *s*.

p. 104, Figure 5
"""
function sample_segmentation(sentence::Array{Int,1}, max_word_length::Int, npylm::PYPContainer)::Array{Int,1}
    N = length(sentence)
    # Initialize to a negative value. If we see the value is negative, we know this box has not been filled yet.
    prob_matrix = fill(-1.0, (N, N))
    # First run the forward filtering
    for t in 1:N
        for k in max(1, t - max_word_length):t
            forward_filtering(sentence, t, k, prob_matrix, npylm)
        end
    end

    segmentation_output = []
    t = N
    i = 0
    # The sentence boundary symbol. Should it be the STOP symbol?
    # OK are those two even the same thing? The START, STOP defined in Corpus and the sentence boundary symbol. Guess I'll have to look at it a bit more then.
    w = STOP

    while t > 0
        # OK I think I get it.
        # It's a bit messy to implement a function just for this `draw` procedure here. Maybe let's just directly write out the procedures anyways.
        # The `zeros` function didn't seem to have worked.
        probabilities = fill(0.0, max_word_length)
        # Keep the w and try out different variations of k, so that different segmentations serve as different context words to the w.
        # Seems that sometimes the max_word_length could be just too big.
        for k in 1:min(max_word_length, t)
            # println("t: $t, k: $k")
            cur_segmentation = sentence[t - k + 1:t]
            cur_context = charseq_to_string(cur_segmentation)
            # Seems that I've created an exception situation. Wonder if there's a better way.
            # cur_word = w
            # if w != STOP
            #     charseq_to_string(w)
            # else
            #     STOP
            # end
            # Need to convert this thing to an array, even though it's just the bigram case. In the trigram case there should be two words instead of one.
            # TODO: In trigram case we need to do something different.
            probabilities[k] = prob(npylm, [cur_context], w)
        end

        # Draw value k from the weights calculated above.
        k = sample(1:max_word_length, Weights(probabilities))
        # This is now the newest word we sampled.
        # The word representation should be converted from the char representation then.
        w = charseq_to_string(sentence[(t - k + 1):t])

        # Update w which indicates the last segmented word.
        # ... Why did I do it two times?
        # w = charseq_to_string(sampled_word)
        # w = sampled_word

        # push!(segmentation_output, sampled_word)
        push!(segmentation_output, w)
        t = t - k
        i += 1
    end

    # The segmented outputs should be properly output in a reverse order.
    return reverse(segmentation_output)
end

"""
Helper function to sample_segmentation. Forward filtering is a part of the algorithm (line 3)

Algorithm documented in section 4.2 of the Mochihashi paper. Equation (7)
Compute α[t][k]
"""
function forward_filtering(sentence::Array{Int,1}, t::Int, k::Int, prob_matrix::Array{Float64,2}, npylm::PYPContainer)::Float64
    # Base case: α[0][n] = 1
    # Another way to do it is to just initialize the array as such. But let's just keep it like this for now, as this is more the responsibility of this algorithm?
    if (t == 0)
        return 1.0
    end
    if (prob_matrix[t,k] >= 0.0)
        return prob_matrix[t,k]
    end
    temp::Float64 = 0.0
    for j = 1:(t - k)
        # The probability here refers to the bigram probability of two adjacent words.
        # Therefore we likely need to convert this thing to word integers first.

        # It's really good that Julia's indexing system makes perfect sense and matches up with the mathematical notation used in the paper perfectly.
        # string_rep_potential_context = charseq_to_string(sentence[(t - k - j + 1):(t - k)], global char_vocab, global word_vocab)
        string_rep_potential_context = charseq_to_string(sentence[(t - k - j + 1):(t - k)])
        # string_rep_potential_word = charseq_to_string(sentence[(t - k + 1):t], global char_vocab, global word_vocab)
        string_rep_potential_word = charseq_to_string(sentence[(t - k + 1):t])
        bigram_prob = prob(npylm, [string_rep_potential_context], string_rep_potential_word)

        temp += bigram_prob * forward_filtering(sentence, (t - k), j, prob_matrix, npylm)
    end

    # Store the final value in the DP matrix.
    prob_matrix[t,k] = temp
    return temp
end

"""
Helper function to forward_filtering

Convert a sequence of characters (represented in Int) to string (represented in Int)
"""
# function charseq_to_string(char_seq::Array{Int,1}, char_vocab::Vocabulary, word_vocab::Vocabulary)::Int
function charseq_to_string(char_seq::Array{Int,1})::Int
    global char_vocab
    global word_vocab
    # First: Convert the (int) character sequence back to their original coherent string
    # Wait, if those two are already global, I wouldn't need to pass them in as arguments anymore.
    string::String = join(map(char_int->get(char_vocab, char_int), char_seq), "")
    # Then: Lookup the string in the word vocab
    string_rep::Int = get(word_vocab, string)
    return string_rep
end


"""
Helper function to sample_segmentation

Function to proportionally draw a k given the prob matrix and indices.

p. 104, Figure 5, line 8

The real thing to do is to actually just draw the thing by:
- Fixing the "dish" (word)
- Changing the context by gradually increasing k
"""
# function draw(sentence::Array{Int, 1}, i::Int, t::Int, w::Int, max_word_length::Int, prob_matrix::Array{Float64,2}, npylm::PYPContainer)
#     probabilities = zeros(Array{Float64,1}, max_word_length)
#     # Keep the w and try out different variations of k, so that different segmentations serve as different context words to the w.
#     for k in 1:max_word_length
#         curr_segmentation = 
#     prob = prob()
#     end
# end

function run_sampler(model::PYPContainer, corpus::Array{Array{Int,1},1}, n_iter::Int, mh_iter::Int)
    n_sentences = length(corpus)
    n_words = sum(length, corpus)
    # Each sentence is turned into a further list of ngrams
    processed_corpus = map(sentence->ngrams(sentence, model.order), corpus)
    for it in 1:n_iter
        println("Iteration $it/$n_iter")

        for sentence in processed_corpus
            # For each ngram that we previously generated from the sentence:
            for ngram in sentence
                # We first remove the customer before sampling it again, because we need to condition the sampling on the premise of all the other customers, minus itself. See Teh et al. 2006 for details.
                if it > 1
                    # The decrement/increment happens by taking all the words in this ngram, minus the last word, as the context.
                    decrement(model, ngram[1:end - 1], ngram[end])
                end
                increment(model, ngram[1:end - 1], ngram[end])
            end
        end

        if it % 10 == 1
            println("Model: $model")
            ll = log_likelihood(model)
            perplexity = exp(-ll / (n_words + n_sentences))
            println("ll=$ll, ppl=$perplexity")
        end

        # Resample hyperparameters every 30 iterations
        # Why 30? I think the original paper had a different approach. Will have to look at that.
        if it % 30 == 0
            println("Resampling hyperparameters")
            acceptance, rejection = resample_hyperparameters(model, mh_iter)
            acceptancerate = acceptance / (acceptance + rejection)
            println("MH acceptance rate: $acceptancerate")
            println("Model: $model")
            ll = log_likelihood(model)
            perplexity = exp(-ll / (n_words + n_sentences))
            println("ll=$ll, ppl=$perplexity")
        end
    end
end

function print_ppl(model::PYPContainer, corpus::Array{Array{Int,1},1})
    n_sentences = length(corpus)
    n_words = sum(length, corpus)
    processed_corpus = map(sentence->ngrams(sentence, model.order), corpus)
    n_oovs = 0
    ll = 0.0

    for sentence in processed_corpus
        for ngram in sentence
            p = prob(model, ngram[1:end - 1], ngram[end])
            if p == 0
                n_oovs += 1
            else
                ll += log(p)
        end
        end
    end
    ppl = exp(-ll / (n_sentences + n_words - n_oovs))
    # ppl = exp(-ll / (n_words - n_oovs))
    println("Sentences: $n_sentences, Words: $n_words, OOVs: $n_oovs")
    println("LL: $ll, perplexity: $ppl")
end

export evaluate;
"""
Load a previously trained model and evaluate it on test corpus.
        corpus
            help=evaluation corpus
            required=true
        model
            help=previously trained model
            required=true
"""
function evaluate(corpus_path, model_path)
    m_in = open(model_path)
    model = deserialize(m_in)
    close(m_in)

    c_in = open(corpus_path)
    # TODO: Deal with the vocabulary in some way so that it's serialized and loaded properly, and hopefully doesn't have any global variable issues.
    evaluation_corpus = read_corpus(c_in, model.vocabulary)
    close(c_in)

    print_ppl(model, evaluation_corpus)
end

end
