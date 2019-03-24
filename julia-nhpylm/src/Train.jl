# include("Corpus.jl")
# include("Trainer.jl")
push!(LOAD_PATH, "./")
import Model
using Serialization

function build_corpus(path)
    corpus = Corpus()
    if isdir(path)
        for file in readdir(path)
            read_file_into_corpus(file, corpus)
        end
    else
        read_file_into_corpus(path, corpus)
    end
    
    return corpus
end

function read_file_into_corpus(path, corpus)
    f = open(path)
    sentences = [[string(c) for c in line if !Base.isspace(c)] for line in readlines(f) if !isempty(line)]
    for sentence_string in sentences
        add_sentence(corpus, sentence_string)
    end
end

function train(corpus_path, split_proportion = 0.9, epochs = 100000, max_word_length = 4)
    corpus = build_corpus(corpus_path)
    dataset = Dataset(corpus, split_proportion)

    println("Number of train sentences $(get_num_train_sentences(dataset))")
    println("Number of dev sentences $(get_num_dev_sentences(dataset))")

    vocabulary = dataset.vocabulary
    # Not sure if this is necessary if we already automatically serialize everything.
    # vocab_file = open(joinpath(pwd(), "npylm.dict"))
    # serialize(vocab_file, vocabulary)
    # close(vocab_file)

    model = Model(dataset, max_word_length)
    set_initial_a(model, HPYLM_a)
    set_initial_b(model, HPYLM_b)
    set_chpylm_beta_stop(model, CHPYLM_β_STOP)
    set_chpylm_beta_pass(model, CHPYLM_β_PASS)

    trainer = Trainer(dataset, model)

    for epoch in 1:epochs
        start_time = time()
        blocked_gibbs_sampling(trainer)
        sample_hyperparameters(trainer)
        sample_lambda(trainer)

        # The accuracy is better after several iterations have been already done.
        if epoch > 3
            update_p_k_given_chpylm(trainer)
        end

        elapsed_time = time() - start_time
        println("Iteration $(epoch), Elapsed time in this iteration: $(elapsed_time)")
        if epoch %10 == 0
            print_segmentations_train(trainer, 10)
            println("Perplexity_dev: $(compute_perplexity_dev(trainer))")
        end
    end

    serialize(model)
end