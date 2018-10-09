__precompile__()
module HPYLM

include("Prior.jl")
include("PYP.jl")
include("Corpus.jl")
include("PYPLM.jl")

function run_sampler(model::PYPLM, corpus::Array{Array{Int,1},1}, n_iter::Int, mh_iter::Int)
    n_sentences = length(corpus)
    n_words = sum(length, corpus)
    processed_corpus = map(sentence -> ngrams(sentence, model.order), corpus)
    for it in 1:n_iter
        println("Iteration $it/$n_iter")

        for sentence in processed_corpus
            for ngram in sentence
                # We first remove the customer before sampling it again, because we need to condition the sampling on the premise of all the other customers, minus itself. See Teh et al. 2006 for details.
                if it > 1
                    decrement(model, ngram[1:end - 1], ngram[end])
                end
                increment(model, ngram[1:end - 1], ngram[end])
            end
        end

        if it % 10 == 1
            println("Model: $model")
            ll = log_likelihood(model)
            perplexity = exp(-ll / (n_words + n_sentences))
            println("ll=$ll, ppl=$perplexity")
        end

        # Resample hyperparameters every 30 iterations
        # Why 30? I think the original paper had a different approach. Will have to look at that.
        if it % 30 == 0
            println("Resampling hyperparameters")
            acceptance, rejection = resample_hyperparameters(model, mh_iter)
            acceptancerate = acceptance / (acceptance + rejection)
            println("MH acceptance rate: $acceptancerate")
            println("Model: $model")
            ll = log_likelihood(model)
            perplexity = exp(-ll / (n_words + n_sentences))
            println("ll=$ll, ppl=$perplexity")
        end
    end
end

export train;
"""
Train the model on training corpus.

Arguments:
        corpus
            help=training corpus
            required=true
        order
            help=order of the model
            arg_type = Int
            required=true
        iter
            help=number of iterations for the model
            arg_type = Int
            required = true
        output
            help=model output path
            required = true
"""
function train(corpus_path, order, iter, output_path)
    # This is the vocabulary of individual characters.
    # Essentially it makes no difference whether we already separate the characters in the input file beforehand, or we do it when reading in the file. Let me just assume plain Chinese text input and do it when reading in the file then.
    char_vocab = Vocabulary()
    word_vocab = Vocabulary()

    println("Reading training corpus")

    f = open(corpus_path)
    training_corpus = read_corpus(f, char_vocab)
    close(f)

    initial_base = Uniform(length(char_vocab))
    model = PYPLM(order, initial_base)

    println("Training model")

    run_sampler(model, training_corpus, iter, 100)

    model.char_vocab = char_vocab
    out = open(output_path, "w")
    serialize(out, model)
    close(out)
end

function print_ppl(model::PYPLM, corpus::Array{Array{Int,1},1})
    n_sentences = length(corpus)
    n_words = sum(length, corpus)
    processed_corpus = map(sentence -> ngrams(sentence, model.order), corpus)
    n_oovs = 0
    ll = 0.0

    for sentence in processed_corpus
        for ngram in sentence
            p = prob(model, ngram[1:end - 1], ngram[end])
            if p == 0
                n_oovs += 1
            else
                ll += log(p)
            end
        end
    end
    ppl = exp(-ll / (n_sentences + n_words - n_oovs))
    # ppl = exp(-ll / (n_words - n_oovs))
    println("Sentences: $n_sentences, Words: $n_words, OOVs: $n_oovs")
    println("LL: $ll, perplexity: $ppl")
end

export evaluate;
"""
Load a previously trained model and evaluate it on test corpus.
        corpus
            help=evaluation corpus
            required=true
        model
            help=previously trained model
            required=true
"""
function evaluate(corpus_path, model_path)
    m_in = open(model_path)
    model = deserialize(m_in)
    close(m_in)

    c_in = open(corpus_path)
    evaluation_corpus = read_corpus(c_in, model.vocabulary)
    close(c_in)

    print_ppl(model, evaluation_corpus)
end

end
