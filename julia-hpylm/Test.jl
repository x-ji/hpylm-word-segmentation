# Let me first try to read in the corpus char by char and see if everything works.
module Test
using Base.Test

include("Prior.jl")
include("PYP.jl")
include("Corpus.jl")
include("PYPLM.jl")

char_vocab = Vocabulary()
f = open("../test.txt")
training_corpus = read_corpus(f, char_vocab)
close(f)

# display(training_corpus)
@test training_corpus == [[3, 4, 5, 6], [7, 8, 9, 3, 10, 11, 12]]

end
