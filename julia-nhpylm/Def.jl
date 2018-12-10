# Beginning of sentence
const BOS = 0
const EOS = 1
# Just realized that these are probably not valid chars. Will need to change to some sort of char type
# This should be fine? AFAIK the text in the corpora are all full-width. Let's see.
# Or well just screw it. Why don't we use the two epsilons or something anyways
const BOW = 'ϵ'
const EOW = 'Ε'

const HPYLM_INITIAL_d = 0.5
const HPYLM_INITIAL_θ = 2.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
const HPYLM_a = 1.0
const HPYLM_b = 1.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
const HPYLM_α = 1.0
const HPYLM_β = 1.0

# const CHPYLM_β_STOP = 4.0
# const CHPYLM_β_PASS = 4.0
# The apparent best values taken from the paper.
const CHPYLM_β_STOP = 0.57
const CHPYLM_β_PASS = 0.85
const CHPYLM_ϵ = 1e-12

# TODO: Expand upon this later, if necessary
const NUM_WORD_TYPES = 1

# In C++ code you can pass in integers by reference. Here I guess you can only create such a struct to hold an integer.
mutable struct IntContainer
    int::Int
end