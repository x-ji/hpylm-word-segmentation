# __precompile__()
# module HPYLM
#
# include("Prior.jl")
# # Currently just lumping everything together instead of exporting structs and functions.
# # using Prior
# include("PYP.jl")
# include("Corpus.jl")
#
# export increment, decrement, PYPLM, Corpus, Vocabulary, Uniform, read_corpus, ngrams, log_likelihood, resample_hyperparameters

import Base.show

# import Base.getindex

 raw"""
 As stated in Yee 2006 (p.987, formula (3)), given a context u, the probability vector for any word following the context u, ``G_{u}``, is
 
 ```math
 G_{u} \sim PY(d_{|u|}, \theta_{|u|}, G_{\pi(u)})
 ```
 
 where ``\pi(u)`` is the suffix of u consisting of all but the earliest word.
 
 Furthermore, ``d_{|u|}`` is the discount and ``\theta_{|u|}`` is the strength; they are both functions of the length ``|u|`` of the context.
 
 Therefore, there are ``2n`` parameters in the model. The *discount* and *strength* parameters, ``d_{|u|}`` and ``\theta_{|u|}``, are shared across all Pitman-Yor processes which have the same ``n`` value.
 
 `PYPLM` serves as the struct that encapsulates several structs which are united around the same `n` value.
 """
mutable struct PYPLM
    "a `PYPPrior` struct, which contains the *discount* and *strength* parameters."
    prior::PYPPrior

    "The order ``n``"
    order::Int

    # It's probably bad design to have two totally different types stand in for one field. Let's see if there's something that can be changed about the design.
    raw"""
    A reference to the `struct` that should be used as the basis of the `backoff` for any `PYP` contained in `models` (i.e. the `struct` on which the ``G_{\pi(u)}`` from the formula above is based).
    - In cases where ``n > 0``, it will be another `PYPLM` struct with an `order` of ``n-1``.
    - In the case where ``n = 0``, it will be the `Uniform` struct, which corresponds to the "global mean vector" mentioned in the paper (p. 988).

    Because two concrete types are both possible, no type annotation is done here. A union type might be a solution though.
    """
    backoff

    "a list of `PYP` structs that have the same ```n`` value but different contexts."
    models::Dict{Array{Int, 1}, PYP}

    """
    This is only useful for serializing the trained model later. NOT something that really functionally belongs to the struct!

    The serialization could be organized better. This can be done later.
    """
    vocabulary::Vocabulary

    # In this project the initial_base will always be `Uniform`.
    function PYPLM(order::Int, initial_base)
        p = new()
        p.prior = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0) # discount = 0.8, theta = 1
        p.order = order
        p.backoff = order == 1 ? initial_base : PYPLM(order - 1, initial_base)
        p.models = Dict{Array{Int, 1}, PYP}()
        return p
    end
end

"This is one possible type for the `base` field of the `PYP` struct. It contains a reference to the `PYPLM` struct of order ``n - 1``, plus a specific context of length ``n - 1``, which will be used to look up the actual `PYP` in the `models` field of the referenced `PYPLM` struct."
struct BackoffBase
    backoff::PYPLM
    ctx::Array{Int, 1}
end

"""
Look up a `PYP` struct contained within the `PYPLM` struct, using the context `ctx`.

If there is no such a `PYP` struct with `ctx` context, a new one will be initialized.

However, note that the `models` field will not be altered in this method! This is because `get` is not supposed to actually cause any changes. The newly initialized `PYP` struct only gets added to `models` in `increment` method.
"""
function get(pyplm::PYPLM, ctx::Array{Int, 1})
    if !haskey(pyplm.models, ctx)
        # Do I really need a BackoffBase construct?... Anyways let me try to replicate vpyp structure first then.
        # When order is 1, we're guaranteed to have pyplm.backoff to be the same as initial_base.
        backoffbase = pyplm.order == 1 ? pyplm.backoff : BackoffBase(pyplm.backoff, ctx[2:end])
        # The whole PYPLM with order shares one prior.
        return PYP(backoffbase, pyplm.prior)
    end
    return pyplm.models[ctx]
end

# Remember that a dish is a word, here the last word in the ngram.
"""
Run `increment` method on the `PYP` struct with context `ctx`, using dish `dish`.

If there is no such a `PYP` struct with `ctx` context, a new one will be initialized via `get` method, and then set in `models` field.
"""
function increment(pyplm::PYPLM, ctx::Array{Int, 1}, dish::Int)
    if !haskey(pyplm.models, ctx)
        pyplm.models[ctx] = get(pyplm, ctx)
    end
    increment(pyplm.models[ctx], dish)
end

"""
Run `decrement` method on the `PYP` struct with context `ctx`, using dish `dish`.

The caller is responsible for ensuring that the `PYP` struct with context `ctx` already exists, and that `dish` already exists in that `PYP`.
"""
function decrement(pyplm::PYPLM, ctx::Array{Int, 1}, dish::Int)
    decrement(pyplm.models[ctx], dish)
end

"""
Run `prob` method on the `PYP` struct with context `ctx`, using dish `dish`, i.e. calculates the conditional probability that the next draw from that `PYP` will be this particular `dish`.

Note that unlike `decrement`, this method **doesn't require neither the `PYP` nor the `dish` to already exist!** The model is capable of calculating the probability for a previously unseen context/dish.
"""
function prob(pyplm::PYPLM, ctx::Array{Int, 1}, dish::Int)
    return prob(get(pyplm, ctx), dish)
end

"Recursively return the log likelihood of the whole model, including those of all the contained PYPs, those of the priors, and those from the backoff structs."
function log_likelihood(pyplm::PYPLM, full::Bool=false)
    return (sum(log_likelihood, values(pyplm.models)) +
            log_likelihood(pyplm.prior) +
            log_likelihood(pyplm.backoff, true))
end

"""
Resample all the parameters.

It first resamples the the parameters in both the priors for this number of `n`, and then it recursively runs this method on all its backoff structs until the backoff model for `n = 0` is reached.
"""
function resample_hyperparameters(pyplm::PYPLM, iteration::Int)
    println("Resampling level $(pyplm.order) hyperparameters")
    (a1, r1) = resample(pyplm.prior, iteration)
    println("resample complete")
    (a2, r2) = resample_hyperparameters(pyplm.backoff, iteration)
    return (a1 + a2, r1 + r2)
end

function Base.show(io::IO, pyplm::PYPLM)
    print(io, "PYPLM(order=$(pyplm.order), #ctx=$(length(pyplm.models)), prior=$(pyplm.prior), backoff=$(pyplm.backoff))")
end

function increment(bb::BackoffBase, dish::Int, initialize::Bool=false)
    increment(bb.backoff, bb.ctx, dish)
end

function decrement(bb::BackoffBase, dish::Int)
    decrement(bb.backoff, bb.ctx, dish)
end

function prob(bb::BackoffBase, dish::Int)
    return prob(bb.backoff, bb.ctx, dish)
end
# end
