include("Def.jl")
include("WHPYLM.jl")
include("CHPYLM.jl")

mutable struct NPYLM
    "The hierarhical Pitman-Yor model for words"
    whpylm::WHPYLM
    "The hierarhical Pitman-Yor model for characters"
    chpylm::CHPYLM

    # I think it's the case that each word has multiple corresponding tables, with each table existing at a different depth. Let's see then.
    prev_depth_at_table_of_token::Dict{UInt, Vector{Vector{Int}}}

    "The cache of WHPYLM G_0?"
    whpylm_g0_cache::Dict{UInt, Float64}
    "The cache of CHPYLM G_0?"
    chpylm_g0_cache::Dict{UInt, Float64}
    lambda_for_types::Vector{Float64}
    "Probability of generating a word of length k from the CHPYLM"
    chpylm_p_k::Vector{Float64}
    max_word_length::UInt
    max_sentence_length::UInt
    "Parameter for the Poisson correction on word length"
    lambda_a::Float64
    "Parameter for the Poisson correction on word length"
    lambda_b::Float64
    "Cache for easier computation"
    hpylm_parent_p_w_cache::Vector{Float64}
    # What chars?
    characters::Vector{Char}

    function NPYLM(max_word_length::UInt, max_sentence_length::UInt, g0::Float64, initial_lambda_a::Float64, initial_lambda_b::Float64, chpylm_beta_stop::Float64, chpylm_beta_pass::Float64)
        npylm = new()

        whpylm = WHPYLM(3)
        chpylm = CHPYLM(g0, max_sentence_length, chpylm_beta_stop, chpylm_beta_pass)

        # Is this really what the original paper proposed? I feel like the author is probably overcomplicating stuffs. Let's see then. For now let me just use one poisson distribution for one iteration of training anyways.
        # TODO: Expand upon word types and use different poisson distributions for different types.
        npylm.lambda_for_types = zeros(Float64, NUM_WORD_TYPES)
        # Currently we use a three-gram model.
        npylm.hpylm_parent_pw_cache = zeros(Float64, 3)
        set_lambda_prior(npylm, initial_lambda_a, initial_lambda_b)

        npylm.max_sentence_length = max_sentence_length
        npylm.max_word_length = max_word_length
        # + 2 because of bow and eow
        # Not sure if this is the most sensible approach with Julia. Surely we can adjust for that.
        npylm.characters = Vector{Char}(max_sentence_length + 2)
        # There are two extra cases where k = 1 and k > max_word_length
        npylm.chpylm_p_k = Vector{Float64}(max_word_length + 2)
        for k in 1:max_word_length + 2
            npylm.chpylm_p_k[k] = 1.0 / (max_word_length + 2)
        end
        return npylm
    end
end