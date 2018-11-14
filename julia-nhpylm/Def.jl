# Beginning of sentence
const BOS = 0
const EOS = 1
const BOW = 0
# Why is this 2? To avoid conflicts with EOS?
const EOW = 2

const HPYLM_INITIAL_d = 0.5
const HPYLM_INITIAL_θ = 2.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
const HPYLM_a = 1.0
const HPYLM_b = 1.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
const HPYLM_α = 1.0
const HPYLM_β = 1.0

const CHPYLM_β_STOP = 4
const CHPYLM_β_PASS = 4
const CHPYLM_ϵ = 1e-12

# TODO: Expand upon this later, if necessary
const NUM_WORD_TYPES = 1