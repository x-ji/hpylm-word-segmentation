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
    d_array::OffsetVector{Float64}
    "Array of concentration parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters."
    θ_array::OffsetVector{Float64}

    #=
    These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)

    Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
    =#
    "For the sampling of discount d"
    a_array::OffsetVector{Float64}
    "For the sampling of discount d"
    b_array::OffsetVector{Float64}
    "For the sampling of concentration θ"
    α_array::OffsetVector{Float64}
    "For the sampling of concentration θ"
    β_array::OffsetVector{Float64}
    #= Constructor =#
    function WHPYLM(order::UInt)
        whpylm = new()
        # depth starts from 0
        whpylm.depth = order - 1
        whpylm.root = PYP{UInt}(0)
        whpylm.root.depth = 0

        whpylm.d_array = OffsetVector{Float64}(undef, 0:0)
        whpylm.θ_array = OffsetVector{Float64}(undef, 0:0)
        whpylm.a_array = OffsetVector{Float64}(undef, 0:0)
        whpylm.b_array = OffsetVector{Float64}(undef, 0:0)
        whpylm.α_array = OffsetVector{Float64}(undef, 0:0)
        whpylm.β_array = OffsetVector{Float64}(undef, 0:0)

        for n in 1:order
            push!(parent(whpylm.d_array), HPYLM_INITIAL_d)
            push!(parent(whpylm.θ_array), HPYLM_INITIAL_θ)
            push!(parent(whpylm.a_array), HPYLM_a)
            push!(parent(whpylm.b_array), HPYLM_b)
            push!(parent(whpylm.α_array), HPYLM_α)
            push!(parent(whpylm.β_array), HPYLM_β)
        end
    end
end

