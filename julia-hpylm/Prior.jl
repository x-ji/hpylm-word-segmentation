# module Prior
# lgamma
using SpecialFunctions

import Base.show

# export SampledPrior, GammaPrior, BetaPrior, PYPPrior

# There are probably built-in functions to do these stuffs in Julia but let me first port the original program faithfully anyways.
"Beta probability density function. Note that this returns the value in natural log."
function beta_pdf(alpha::Float64, beta::Float64, x::Float64)
    return (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta) +
            (alpha - 1) * log(x) + (beta - 1) * log(1 - x))
end

"Gamma probability density function. Note that this returns the value in natural log."
function gamma_pdf(shape::Float64, scale::Float64, x::Float64)
    return (- shape * log(scale) - lgamma(shape) +
            (shape - 1) * log(x) - x/scale)
end

"""
The base type for all prior types that can be sampled

Note that `tied_distributions`, which contains all the distributions (e.g. `PYP` structs) which get their parameters from this prior, should really be a field that's present in all prior types. Unfortunately as of Julia 0.6, field in abstract type is not supported yet. Thus it is specified in each of the concrete `struct` definitions.
"""
abstract type SampledPrior end

"A gamma distribution acting as a prior."
mutable struct GammaPrior <: SampledPrior
    tied_distributions
    shape::Float64
    scale::Float64

    """
    The current value of the parameter of interest (in the case of this project, discount + strength).

    I think the reason why it is set as discount + strength is that in this way we can ensure discount + strength >= 0, i.e. strength > -discount!
    """
    x::Float64
    function GammaPrior(shape, scale, x)
        p = new()
        p.shape = shape
        p.scale = scale
        p.x = x
        p.tied_distributions = []
        return p
    end
end

"Calculate the likelihood of the current strength + discount value given the current gamma prior"
function log_likelihood(p::GammaPrior)
    return gamma_pdf(p.shape, p.scale, p.x)
end

"Get the current value of the parameter of interest (in the case of this project, discount + strength)"
function get_parameters(p::GammaPrior)
    # This is really an awkward approach...
    return (p.x,)
end

"Set the current value of the parameter of interest (in the case of this project, discount + strength)"
function set_parameters(p::GammaPrior, params::Tuple{Float64, Float64})
    p.x, = params
end

"Re-sample the parameter of interest (discount + strength). Note that the probability density function is the same as the one used in `proposal_log_likelihood`, of course."
function sample_parameters(p::GammaPrior)
    # With the shape set to 1, it's an exponential distribution.
    # The mean of a gamma distribution is shape * scale. So by setting shape to 1, the mean is x. It's centered around x.
    dist = Gamma(1, p.x)
    p.x = rand(dist)
    if p.x <= 0
        p.x = 1e-12
    end
end

"""
This is the proposal density function used in the MH sampling process. We generate a new parameter value depending on the old parameter value.

Here we just simply use a Gamma distribution centered around the old parameter value to generate a new value.
"""
function proposal_log_likelihood(p::GammaPrior, old_params, new_params)
    return gamma_pdf(1.0, old_params[1], new_params[1])
end

function Base.show(io::IO, p::GammaPrior)
    print(io, "GammaPrior(x=$(p.x) ~ Gamma($(p.shape), $(p.scale)) | nties=$(length(p.tied_distributions))")
end

"A beta distribution acting as a prior."
mutable struct BetaPrior <: SampledPrior
    tied_distributions
    alpha::Float64
    beta::Float64

    "The current value of the parameter of interest (in the case of this project, discount)"
    x::Float64
    function BetaPrior(alpha, beta, x)
        p = new()
        p.alpha = alpha
        p.beta = beta
        p.x = x
        p.tied_distributions = []
        return p
    end
end

"Calculate the likelihood of the current discount value given the current beta prior"
function log_likelihood(p::BetaPrior)
    return beta_pdf(p.alpha, p.beta, p.x)
end

"Get the current value of the parameter of interest (in the case of this project, discount)"
function get_parameters(p::BetaPrior)
    return (p.x,)
end

"Set the current value of the parameter of interest (in the case of this project, discount)"
function set_parameters(p::BetaPrior, params::Tuple{Float64, Float64})
    p.x, = params
end

"Sample a new discount value. If it turned out to be <= 0 or >= 1, just set it to 0.5. Note that the probability density function is the same as the one used in `proposal_log_likelihood`, of course."
function sample_parameters(p::BetaPrior)
    # This formulation ensures that the mean is x. Though I'm not totally sure why the author chose to set \alpha to 10 instead of 5 for example.
    dist = Beta(10, 10*(1-p.x)/p.x)
    p.x = rand(dist)
    if p.x <= 0 || p.x >= 1
        p.x = 0.5
    end
end

"""
This is the proposal density function used in the MH sampling process. We generate a new parameter value depending on the old parameter value.

The mean of this distribution is x. Though I'm not totally sure about the choice of alpha parameter as 10.
"""
function proposal_log_likelihood(p::BetaPrior, old_params, new_params)
    return beta_pdf(10.0, 10*(1-old_params[1])/old_params[1], new_params[1])
end

function Base.show(io::IO, p::BetaPrior)
    print(io, "BetaPrior(x=$(p.x) ~ Beta($(p.alpha), $(p.beta)) | nties=$(length(p.tied_distributions))")
end

"""
A convenience struct that contains one `BetaPrior` and one `GammaPrior`. Each `PYPContainer` struct contains one `PYPPrior` struct.
"""
mutable struct PYPPrior <: SampledPrior
    tied_distributions
    d_prior::BetaPrior
    s_prior::GammaPrior
    function PYPPrior(d_alpha::Float64, d_beta::Float64, s_shape::Float64, s_scale::Float64, discount::Float64, strength::Float64)
        p = new()
        p.d_prior = BetaPrior(d_alpha, d_beta, discount)
        p.s_prior = GammaPrior(s_shape, s_scale, discount + strength)
        p.tied_distributions = []
        return p
    end
end

function discount(p::PYPPrior)
    return p.d_prior.x
end

function strength(p::PYPPrior)
    return p.s_prior.x - p.d_prior.x
end

"""
Calculate the log likelihood of the current strength and discount, considered together.

The s_prior's parameter actually represents s + d, thus the minus operation? Or is it simply an error...
"""
function log_likelihood(p::PYPPrior)
    return log_likelihood(p.s_prior) - log_likelihood(p.d_prior)
end

"Return discount and strength as a tuple"
function get_parameters(p::PYPPrior)
    return (p.d_prior.x, p.s_prior.x)
end

"Set discount and strength from a given tuple"
function set_parameters(p::PYPPrior, params::Tuple{Float64, Float64})
    p.d_prior.x, p.s_prior.x = params
end

"Resample both the discount and the strength"
function sample_parameters(p::PYPPrior)
    sample_parameters(p.d_prior)
    sample_parameters(p.s_prior)
end

"The proposal density function for the PYPPrior is simply the sum of proposal density functions of both the Gamma prior and the Beta prior."
function proposal_log_likelihood(p::PYPPrior, old_params::Tuple{Float64, Float64}, new_params::Tuple{Float64, Float64})
    return (proposal_log_likelihood(p.d_prior, (old_params[1],), (new_params[1],)) +
            proposal_log_likelihood(p.s_prior, (old_params[2],), (new_params[2],)))
end

function Base.show(io::IO, p::PYPPrior)
    print(io, "PYPPrior(discount=$(discount(p)), strength=$(strength(p)) | discount ~ Beta($(p.d_prior.alpha), $(p.d_prior.beta)); strength + discount ~ Gamma($(p.s_prior.shape), $(p.s_prior.scale)) | nties = $(length(p.tied_distributions)))")
end

function tie(p::SampledPrior, distribution)
    # Memory cost > 100MB. I have no idea what's happening in this one. I think these should also just be pointers to PYPs eh. Let's see then.
    push!(p.tied_distributions, distribution)
end

"Get the full log likelihood, i.e. not only that of the prior itself, but also all the distributions (PYPs) that have their discount and strength tied to the prior"
function full_log_likelihood(p::SampledPrior)
    return sum(log_likelihood, p.tied_distributions) + log_likelihood(p)
end

"""
Metropolis-Hastings resampling of all the current parameter values obtained from this prior, with `iteration` number of iterations.

Recall that we have:

``A(x'|x) / A(x|x') = P(x')g(x|x') / P(x)g(x'|x)``

or,

``A(x'|x) / A(x|x') = P(x')Q(x|x') / P(x)Q(x'|x)``

The acceptance probability is

``A(x'|x) = min(1, P(x')g(x|x') / P(x)g(x'|x))``
"""
function resample(p::SampledPrior, iteration)
    accept_reject = [0, 0]
    # This is P(x)
    old_ll = full_log_likelihood(p)
    for _ in 1:iteration
        old_parameters = get_parameters(p)
        sample_parameters(p)
        new_parameters = get_parameters(p)
        # This is P(x')
        new_ll = full_log_likelihood(p)
        # This is Q(x|x')
        old_lq = proposal_log_likelihood(p, new_parameters, old_parameters)
        # This is Q(x'|x)
        new_lq = proposal_log_likelihood(p, old_parameters, new_parameters)

        # Addition/Subtraction is performed because they're log values. Otherwise it would be multiplication/division.
        # This is P(x')Q(x|x') / P(x)Q(x'|x)
        log_acc = new_ll + old_lq - old_ll - new_lq

        # If the log result is > 0, the non-log result is > 1. The proposal should always be accepted.
        # Otherwise, there is a possibility proportional to the non-log result to accept the proposal.
        if log_acc > 0 || rand() < exp(log_acc)
            accept_reject[1] += 1
            old_ll = new_ll
        else # Proposal rejected
            accept_reject[2] += 1
            set_parameters(p, old_parameters) # Put the old set of parameters back.
        end
    end

    return accept_reject
end

function parameters(p::SampledPrior)
    return get_parameters(p)
end

function parameters(p::SampledPrior, params::Tuple{Float64, Float64})
    set_parameters(p, params)
end
# end
