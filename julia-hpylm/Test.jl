push!(LOAD_PATH, "./")
import HPYLM

models = ["ptbmodel", "wiki2model", "brownmodel", "lobmodel", "sotumodel"]
corpora = ["../data/ptb/test.txt", "../data/wikitext-2/wiki.test.tokens", "../data/brown/brown-test.txt", "../data/LOB/LOB_COCOA/test.txt", "../data/sotu/sotu-test-utf8.txt",]

for model in models
    for corpus in corpora
        println("Model: $model, Corpus: $corpus")
        HPYLM.evaluate(corpus, model)
    end
end
