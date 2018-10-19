"Represent the uniform distribution. Used as the fallback distribution in the case of `n = 0`."
mutable struct UniformDist
    K::Int
    count::Int
    function UniformDist(K::Int)
        d = new()
        d.K = K
        d.count = 0
        return d
    end
end

function increment(d::UniformDist, k, initialize)
    d.count += 1
end

function decrement(d::UniformDist, k)
    d.count -= 1
end

function prob(d::UniformDist, k::Int)
    if k >= d.K
        return 0
    end
    return 1.0 / d.K
end

function log_likelihood(d::UniformDist, full)
    return -(d.count * log(d.K))
end

function resample_hyperparameters(d::UniformDist, iteration)
    return (0, 0)
end

function Base.show(io::IO, d::UniformDist)
    print(io, "UniformDist(K=$(d.K), count = $(d.count))")
end
