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
    char_seq = string_to_charseq(dish)
    # Implementing the infinite gram model turned out to be a bit involved to implement. Guess I'm left without any choice but to just break down the characters into 3-grams (with padding) and simply multiply the 3-gram probabilities together then. Let's see.
    # TODO: Actually implement the infinite gram model.
    char_hpylm_prob = 1.0
    char_ngrams = ngrams(char_seq, pyp.base.order)
    for ngram in char_ngrams
        char_hpylm_prob *= prob(pyp.base, ngram[1:end-1], ngram[end])
    end
    # Now this just became incredibly slow since we're doing a lot of operations inside of another nested model, instead of just falling back upon an Uniform model, IMO. Eh.
    # Originally this thing wasn't particularly fast either. But definitely not this slow. Let's see how we can actually get things done then.
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

    function PYPContainer(order::Int, initial_base, is_for_words::Bool = true)
        p = new()
        p.prior = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0) # discount = 0.8, theta = 1
        p.order = order
        # Note that if this PYPContainer is a word PYPContainer and order == 1, the initial base will be a char PYPContainer.
        p.backoff = order == 1 ? initial_base : PYPContainer(order - 1, initial_base, is_for_words)
        p.models = Dict{Array{Int,1},PYP}()
        p.is_for_words = is_for_words
        return p
    end
end

"This is one possible type for the `base` field of the `PYP` struct. It contains a reference to the `PYPContainer` struct of order ``n - 1``, plus a specific context of length ``n - 1``, which will be used to look up the actual `PYP` in the `models` field of the referenced `PYPContainer` struct."
struct BackoffBase
    # This naming makes much more sense. This is the PYPContainer for the backing off of the PYP that contains this BackoffBase.
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
        # The special treatments concerning the transition from word HPYLM to char HPYLM are done in `increment`, `decrement`, and `prob` functions.
        backoffbase = pyp_container.order == 1 ? pyp_container.backoff : BackoffBase(pyp_container.backoff, ctx[2:end])
        # The whole PYPContainer with the same order shares one prior.
        return PYP(backoffbase, pyp_container.prior)
    end
    return pyp_container.models[ctx]
end

"""
Helper function to increment

Convert a string (represented in Int) to its sequence of chars (represented in Int)
"""
function string_to_charseq(str::Int)::Array{Int,1}
    global char_vocab
    global word_vocab
    # First: Convert the string from int to its original form
    word::String = get(word_vocab, str)
    # Then: Look up the characters that constitute the word one by one
    # Seems that somehow with the call to `string`, the String is regarded as an AbstractString, and I'll need to preemptively convert the String to an Array with the `collect` method. Eh.
    return map(char -> get(char_vocab, string(char)), collect(word))
end

"""
Helper function

Convert a sequence of characters (represented in Int) to string (represented in Int)
"""
function charseq_to_string(char_seq::Array{Int,1})::Int
    global char_vocab
    global word_vocab
    # First: Convert the (int) character sequence back to their original coherent string
    string::String = join(map(char_int->get(char_vocab, char_int), char_seq), "")
    # Then: Lookup the string in the word vocab
    string_rep::Int = get(word_vocab, string)
    return string_rep
end

# Remember that a dish is a word, here the last word in the ngram.
"""
Run `increment` method on the `PYP` struct with context `ctx`, using dish `dish`.

If there is no such a `PYP` struct with `ctx` context, a new one will be initialized via `get` method, and then set in `models` field.
"""
function increment(pyp_container::PYPContainer, ctx::Array{Int,1}, dish::Int)
    # Special treatment for the transition.
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
    if pyp_container.is_for_words && pyp_container.order == 1
        # In this case, the `get` method returns the PYP of G_1, which doesn't have any context to speak of whatsoever. This is to say, at G_1 level there's only *one single PYP object* contained in the `pyp_container.models` field.
        # We need a special prob method. Maybe there are better ways to do this but let me go through with this implementation at first.
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
