# Beginning of sentence
# Since 0 and 1 are not Char/String in Julia, might be useful to add some unique representations as well.
# Use alpha and omega then.
const BOS_CHAR = 'Α'
const EOS_CHAR = 'Ω'

# Since they are technically words, we should also store their hashed representations.
# Might well hash the string versions of them to ensure that their hash values don't conflict with those of the actual words.
const BOS = hash(string(BOS_CHAR))
const EOS = hash(string(EOS_CHAR))

# Just realized that these are probably not valid chars. Will need to change to some sort of char type
# This should be fine? AFAIK the text in the corpora are all full-width. Let's see.
# Lower-case alpha and omega
const BOW = 'α'
const EOW = 'ω'

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

# In C++ code you can pass in integers by reference. Here I guess you can only create such a struct to hold an integer.
# Should remove this later and adopt a return-value-oriented way.
mutable struct IntContainer
    int::Int
end