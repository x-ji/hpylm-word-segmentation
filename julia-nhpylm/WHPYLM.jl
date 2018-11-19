include("Def.jl")
include("PYP.jl")

"""
Hierarchical Pitman-Yor process for words
"""
mutable struct WHPYLM{T} <: HPYLM{T}
    #= All the fields are "inherited" from HPYLM. Or, to put it another way, unlike CHPYLM, WHPYLM doesn't have its own new fields. =#
    "Root PYP which has no context"
    root::PYP{T}
    "Depth of the whole HPYLM"
    depth::UInt
    "Base probability for 0-grams, i.e. G_0(w)"
    G_0::Float64
    "Array of discount parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    d_array::Vector{Float64}
    "Array of concentration parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    θ_array::Vector{Float64}

    #=
    These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)

    Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
    =#
    "For the sampling of discount d"
    a_array::Vector{Float64}
    "For the sampling of discount d"
    b_array::Vector{Float64}
    "For the sampling of concentration θ"
    α_array::Vector{Float64}
    "For the sampling of concentration θ"
    β_array::Vector{Float64}

    #= Constructor =#
    function WHPYLM(order::UInt)
        whpylm = new()
        # depth starts from 0
        whpylm.depth = order - 1
        whpylm.root = PYP{UInt}(0)
        whpylm.root.depth = 0

        for n in 1:order
            push!(whpylm.d_array, HPYLM_INITIAL_d)
            push!(whpylm.θ_array, HPYLM_INITIAL_θ)
            push!(whpylm.a_array, HPYLM_a)
            push!(whpylm.b_array, HPYLM_b)
            push!(whpylm.α_array, HPYLM_α)
            push!(whpylm.β_array, HPYLM_β)
        end
    end
end

