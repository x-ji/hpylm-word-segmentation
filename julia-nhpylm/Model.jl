# Put all necessary imports here.
__precompile__()
module NHPYLM
using Serialization

include("Def.jl")
include("CType.jl")
include("WType.jl")
include("Sentence.jl")
include("Corpus.jl")
include("PYP.jl")
include("HPYLM.jl")
include("CHPYLM.jl")
include("WHPYLM.jl")
include("NPYLM.jl")
include("Sampler.jl")

"""
This is the struct that will serve as a container for the whole NHPYLM. it will be serialized after training.
"""
struct Model
    npylm::NPYLM
    sampler::Sampler
    function Model(dataset::Dataset, max_word_length::Int)
        max_sentence_length = dataset.max_sentence_length
        # The G_0 probability for the character HPYLM, which depends on the number of different characters in the whole corpus.
        chpylm_G_0 = 1.0 / get_num_characters(dataset.vocabulary)

        # Need to do this because `Model` is immutable
        npylm = NPYLM(max_word_length, max_sentence_length, chpylm_G_0, 4.0, 1.0, CHPYLM_β_STOP, CHPYLM_β_PASS)
        sampler = Sampler(npylm, max_word_length, max_sentence_length)
        return new(npylm, sampler)
    end

    function Model(dataset::Dataset, max_word_length::Int, initial_a, initial_b, chpylm_β_stop, chpylm_β_pass)
        model = new()
        max_sentence_length = dataset.max_sentence_length
        chpylm_G_0 = 1.0 / get_num_characters(dataset.vocabulary)
        model.npylm = NPYLM(max_word_length, max_sentence_length, chpylm_G_0, initial_a, initial_b, chpylm_β_stop, chpylm_β_pass)
        model.sampler = Sampler(model.npylm, max_word_length, max_sentence_length)
        return model
    end

    # Try to restore this thing from a file. Not sure if I'll need a particular method for that.
    # function Model(file::IOStream)
    #     model = new()
    #     model.npylm = NPYLM()

    #     return model
    # end
end

function get_max_word_length(model::Model)
    return model.npylm.max_word_length
end

function set_initial_a(model::Model, initial_a::Float64)
    model.npylm.λ_a = initial_a
    sample_λ_with_initial_params(model.npylm)
end

function set_initial_b(model::Model, initial_b::Float64)
    model.npylm.λ_b = initial_b
    sample_λ_with_initial_params(model.npylm)
end

function set_chpylm_beta_stop(model::Model, stop::Float64)
    model.npylm.chpylm.beta_stop = stop
end

function set_chpylm_beta_pass(model::Model, pass::Float64)
    model.npylm.chpylm.beta_pass = pass
end

function segment_sentence(model::Model, sentence_string::UTF32String)::OffsetVector{UTF32String}
    extend_capacity(model.sampler, model.npylm.max_word_length, length(sentence_string))
    extend_capacity(model.npylm, length(sentence_string))
    segmented_sentence = OffsetVector{UTF32String}(undef, 0:-1)
    sentence = Sentence(sentence_string)
    segment_lengths = viterbi_decode(model.sampler, sentence)

    # I don't even think there's the need to run this method anyways, since all we need is the vector of words eventually.
    # split_sentence(sentence, segment_lengths)

    # Skip the first two BOS in the sentence.
    # start_index = 3

    start_index = 1
    # for (index, length) in enumerate(segment_lengths)
    for length in segment_lengths
        word = sentence_string[start_index, start_index + length - 1]
        push!(segmented_sentence, word)
        start_index += length
    end

    return segmented_sentence
end

"Compute the log forward probability of any sentence given the whole NHPYLM model"
function compute_log_forward_probability(model::Model, sentence_string::UTF32String, with_scaling::Bool)
    extend_capacity(model.sampler, model.npylm.max_word_length, length(sentence_string))
    extend_capacity(model.npylm, length(sentence_string))
    sentence = Sentence(sentence_string)
    return compute_log_forward_probability(model.sampler, sentence, with_scaling)
end

# Actually I'm not sure if we really need such a complicated Trainer class. Let's first go on though.
"This struct contains everything needed for the training process"
mutable struct Trainer
    rand_indices_train::Vector{Int}
    rand_indices_dev::Vector{Int}
    dataset::Dataset
    vocabulary::Vocabulary
    model::Model
    "These tables are used when we generate words randomly from the CHPYLM, in the `sample_next_char_from_chpylm_given_context` function."
    chpylm_sampling_probability_table::Vector{Float64}
    chpylm_sampling_id_table::Vector{Char}
    always_accept_new_segmentation::Bool
    "Indicates whether the sentence at this index has already been added to the CHPYLM. If yes, in iterations > 2 we'd need to remove the sentence from the CHPYLM and add it again."
    added_to_chpylm_train::Vector{Bool}
    "If we don't always accept new segmentations, some segmentations might be rejected."
    num_segmentation_rejections::Int
    num_segmentation_acceptances::Int
    function Trainer(dataset::Dataset, model::Model, always_accept_new_segmentation::Bool=true)
        trainer = new()
        trainer.dataset = dataset
        trainer.model = model
        trainer.vocabulary = dataset.vocabulary
        # Need extra space for EOS
        trainer.chpylm_sampling_probability_table = Vector{Float64}(undef, get_num_characters(trainer.vocabulary) + 1)
        trainer.chpylm_sampling_id_table = Vector{Char}(undef, get_num_characters(trainer.vocabulary) + 1)
        trainer.added_to_chpylm_train = fill(false, (length(dataset.train_sentences)))
        trainer.rand_indices_train = Vector{Int}(undef, length(dataset.train_sentences))
        for i in 1:length(dataset.train_sentences)
            trainer.rand_indices_train[i] = i
        end
        trainer.rand_indices_dev = Vector{Int}(undef, length(dataset.dev_sentences))
        for i in 1:length(dataset.dev_sentences)
            trainer.rand_indices_dev[i] = i
        end
        trainer.always_accept_new_segmentation = always_accept_new_segmentation
        trainer.num_segmentation_rejections = 0
        trainer.num_segmentation_acceptances = 0
        return trainer
    end
end

function sample_hyperparameters(trainer::Trainer)
    sample_hyperparameters(trainer.model.npylm)
end

"""
Sample lambda values for different types of characters.
    
For example, puncutation marks, alphabets, Chinese ideographs are all different types of characters.

Each type would get its own average word length correction with a different lambda value.
"""
function sample_lambda(trainer::Trainer)
    a_array = zeros(Float64, WORDTYPE_NUM_TYPES)
    b_array = zeros(Float64, WORDTYPE_NUM_TYPES)
    word_ids::Set{UInt} = Set()
    for t in 1:WORDTYPE_NUM_TYPES
        a_array[t] = trainer.model.npylm.λ_a
        b_array[t] = trainer.model.npylm.λ_b
    end
    # Get all sentences in the training set.
    for sentence in trainer.dataset.train_sentences
        # Go through each word in the sentence, excluding the BOS and EOS tokens.
        for index in 2:sentence.num_segments - 2
            word::UTF32String = get_nth_word_string(sentence, index)
            word_id::UInt = get_nth_word_id(sentence, index)
            word_length::Int = get_nth_segment_length(sentence, index)
            if word_length > trainer.model.npylm.max_word_length
                continue
            end

            # If the word hasn't been added to the set of known words yet, add it.
            if !in(word_id, word_ids)
                # Get the tablegroups that correspond to this word in the root of the WHPYLM. Essentially we're just trying to count how frequent this word appeared.
                # IMO this word should always be present in the root of the WHPYLM. If not then it's a bug. Anyways the [] there is just a failsafe measure.
                tablegroups = get(trainer.model.npylm.whpylm.root.tablegroups, word_id, [])
                num_tablegroups = length(tablegroups)
                # TODO: Need to properly detect the word type.
                t = detect_word_type(word)
                a_array[t] += num_tablegroups * word_length
                b_array[t] += num_tablegroups
                push!(word_ids, word_id)
            end
        end
        for t in 1:WORDTYPE_NUM_TYPES
            dist = Gamma(a_array[t], 1 / b_array[t])
            trainer.model.npylm.λ_for_types[t] = rand(dist)
        end
    end
end

"""
This function tries to generate a word randomly from the CHPYLM. Used by the function `update_p_k_given_chpylm`.

`skip_eow` means that EOW shouldn't be generated as the next char. This applies when there is only BOW in the current word so far.
"""
function sample_next_char_from_chpylm_given_context(trainer::Trainer, context_chars::OffsetVector{Char}, context_length::Int, sample_t::Int, skip_eow::Bool)
    prob_sum = 0.0
    chpylm = trainer.model.npylm.chpylm
    table_index = 1
    all_characters = trainer.vocabulary.all_characters
    num_characters = length(all_characters)
    for c in all_characters
        # context_begin: 0, context_end: length - 1
        p_w = compute_p_w_given_h(chpylm, c, context_chars, 0, context_length - 1)
        prob_sum += p_w
        trainer.chpylm_sampling_probability_table[table_index] = p_w
        trainer.chpylm_sampling_id_table[table_index] = c
        table_index += 1
    end

    # Also record EOW as a probable character to be sampled.
    if !skip_eow
        p_w = compute_p_w_given_h(chpylm, EOW, context_chars, 0, context_length - 1)
        prob_sum += p_w
        trainer.chpylm_sampling_probability_table[table_index] = p_w
        trainer.chpylm_sampling_id_table[table_index] = EOW
    end

    # Sample one character from the table.
    return sample(trainer.chpylm_sampling_id_table, Weights(trainer.chpylm_sampling_probability_table))
end

"""
This function updates the cache of the probability of sampling a word of length k from the CHPYLM.

As mentioned in Section 4.3 of the paper, a Monte Carlo method is employed to generate words randomly from the CHPYLM so that empirical estimates of p(k|chpylm) can be obtained.
"""
function update_p_k_given_chpylm(trainer::Trainer, num_samples::Int = 20000, early_stopping_threshold::Int = 10)
    max_word_length = get_max_word_length(trainer.model) + 1
    p_k_chpylm = trainer.model.npylm.p_k_chpylm
    # Do you mean num_characters. Eh.
    # It's 1 longer than the original max_word_length, probably we have 0 in order to incorporate the possibility of getting length 0 word?
    # This array keeps track of total numbers of words of length k.
    # max_word_length + 1 because also a special case of k > max_word_length needs to be tracked?
    # Note that we need to provide a type argument to zeros in this case.
    num_words_of_length_k::OffsetVector{Int} = zeros(Int, 0:max_word_length)
    for i in 0:max_word_length
        p_k_chpylm[i] = 0.0
    end

    wrapped_chars = OffsetVector{Char}(undef, 0:max_word_length + 2)
    num_words_sampled = 0
    for itr in 1:num_samples
        wrapped_chars[0] = BOW

        # Keeps track of the actual word length
        cur_word_length = 0
        for j in 0:max_word_length - 1
            # If we're only at the beginning of the sentence we shouldn't sample an EOW, because that would result in the "word" containing no characters at all.
            skip_eow = (j == 0) ? true : false
            next_char = sample_next_char_from_chpylm_given_context(trainer, wrapped_chars, j + 1, j + 1, skip_eow)
            wrapped_chars[j + 1] = next_char
            # EOW means the word is completely sampled.
            if next_char == EOW
                break
            end
            cur_word_length += 1
        end

        # In this case we just sampled an empty word, i.e. <BOW><EOW>. It cannot be used. Continue to the next round of sampling.
        if cur_word_length == 0
            continue
        end

        @assert cur_word_length <= max_word_length
        num_words_of_length_k[cur_word_length] += 1

        # If all possible lengths have enough data generated, we can terminate the sampling early.
        if itr % 100 == 0
            can_stop = true
            for k in 1:max_word_length
                if num_words_of_length_k[k] < early_stopping_threshold
                    can_stop = false
                    break
                end
            end
            if can_stop
                break
            end
        end
    end

    for k in 1:max_word_length
        # Put in a Laplace smoothing over the final figures. Though seems that the divisor doesn't need this treatment anyways.
        # p_k_chpylm[k] = (num_words_of_length_k[k] + 1) / (num_words_sampled + max_word_length + 1)
        p_k_chpylm[k] = (num_words_of_length_k[k] + 1) / (num_words_sampled + max_word_length)
        @assert p_k_chpylm[k] > 0
    end
end

function blocked_gibbs_sampling(trainer::Trainer)
    # Yeah I think we're not doing any segmentation in the first round at all. Segmentations only start from the second round. So the behavior is normal.
    # Still then the problem is why on only certain sentences they try to remove EOS twice from the table. Fucking hell this just simply doesn't make the least bit of sense whatsoever let's just go on and see then. Such a load of total jokes inded thoasdf otasd foatsdfas taosdfast aosdfasdftreadsfasdftadsfasdf total absdfoatr toasdlfasetrdf.
    # temp_sentence = trainer.dataset.train_sentences[1]
    # println("In blocked_gibbs_sampling, temp_sentence is $temp_sentence, temp_sentence.num_segments is $(temp_sentence.num_segments), temp_sentence.segment_lengths is $(temp_sentence.segment_lengths) ")
    num_sentences = length(trainer.dataset.train_sentences)
    max_sentence_length = trainer.dataset.max_sentence_length

    # TODO: ... Why don't you just shuffle the array of sentences itself instead of this seemingly extraneous array of indices?
    shuffle!(trainer.rand_indices_train)

    # Update model parameters
    for step in 1:num_sentences
        sentence_index = trainer.rand_indices_train[step]
        sentence = trainer.dataset.train_sentences[sentence_index]

        if sentence.supervised
            # Remove the segmentation and add it again, so that the seating arrangements can be updated.
            if trainer.added_to_chpylm_train[sentence_index] == true
                for n in 2:sentence.num_segments - 1
                    remove_customer_at_index_n(trainer.model.npylm, sentence, n)
                end
            end
            # Because this is supervised data, i.e. guaranteed to be the true segmentation, we don't need to resample the sentence at all.
            for n in 2:sentence.num_segments - 1
                add_customer_at_index_n(trainer.model.npylm, sentence, n)
            end
            trainer.added_to_chpylm_train[sentence_index] = true
            continue
        else
            # TODO: I thought this has more to do with the iteration of sampling? Do we really need such a mechanism anyways. But where is the iteration number in the first place eh.
            if trainer.added_to_chpylm_train[sentence_index] == true
                old_segment_lengths = OffsetVector{Int}(undef, 0:max_sentence_length + 2)
                num_old_segments = 0
                old_log_p_s = 0.0
                new_log_p_s = 0.0

                # Wait, why is this thing triggered in the first round already. Even this doesn't seem to make sense.
                for n in 2:sentence.num_segments - 1
                    # println("In blocked_gibbs_sampling, n is $n, sentence is $sentence, sentence.num_segments is $(sentence.num_segments), sentence.segment_lengths is $(sentence.segment_lengths) ")
                    remove_customer_at_index_n(trainer.model.npylm, sentence, n)
                end

                # We need to later decide by some criteria whether to accept the new segmentation or just keep the old one.
                if trainer.always_accept_new_segmentation == false
                    num_old_segments = get_num_segments_without_special_tokens(sentence)
                    for i in 0:num_old_segments - 1
                        # We save the old segmentation but get rid of the BOS and EOS tokens
                        # Two BOS in the beginning.
                        old_segment_lengths[i] = sentence.segments[i + 2]
                    end
                    old_log_p_s = compute_log_probability_of_sentence(trainer.model.npylm, sentence)
                end

                # Produce the new segmentation
                new_segment_lengths = blocked_gibbs_segment(trainer.model.sampler, sentence, true)
                # println("new_segment_lengths is $new_segment_lengths")
                split_sentence(sentence, new_segment_lengths)

                # TODO: There might be a way to avoid performing the check twice? Using a single Sentence struct to hold all these stuffs is quite a bit restrictive.
                if trainer.always_accept_new_segmentation == false
                    new_log_p_s = compute_log_probability_of_sentence(trainer.model.npylm, sentence)
                    # When the log probability of the new segmentation is lower, accept the new segmentation only with a certain probability
                    bernoulli = min(1.0, exp(new_log_p_s - old_log_p_s))
                    r = rand(Float64)
                    if bernoulli < r
                        split_sentence(sentence, old_segment_lengths, num_old_segments)
                        trainer.num_segmentation_rejections += 1
                    else
                        trainer.num_segmentation_acceptances += 1
                    end
                end
            end

            # Put in the new segmentation results
            for n in 2:sentence.num_segments - 1
                add_customer_at_index_n(trainer.model.npylm, sentence, n)
            end
            trainer.added_to_chpylm_train[sentence_index] = true
        end
    end
    @assert trainer.model.npylm.whpylm.root.ntables <= get_num_customers(trainer.model.npylm.chpylm)
end

# TODO: Summarize the difference between the usgae of the Viterbi algorithm and the original blocked sampler.
"Compute the perplexity based on optimal segmentation produced by the Viterbi algorithm"
function compute_perplexity(trainer::Trainer, sentences::Vector{Sentence})
    num_sentences = length(sentences)
    if num_sentences == 0
        return 0.0
    end
    sum = 0.0

    for s in sentences
        # Create a copy so that no interference occurs.
        sentence = Sentence(s.sentence_string)
        segment_lengths = viterbi_decode(trainer.model.sampler, sentence)
        split_sentence(sentence, segment_lengths)
        # Why - 2 not - 3 though? EOS still needs to be taken into account in perplexity computation I guess?
        sum += compute_log_probability_of_sentence(trainer.model.npylm, sentence) / sentence.num_segments - 2
    end

    ppl = exp(-sum / num_sentences)
    return ppl
end

function compute_perplexity_train(trainer::Trainer)
    return compute_perplexity(trainer, trainer.dataset.train_sentences)
end

function compute_perplexity_dev(trainer::Trainer)
    return compute_perplexity(trainer, trainer.dataset.dev_sentences)
end

function compute_log_likelihood(trainer::Trainer, sentences::OffsetVector{Sentence})
    num_sentences = length(sentences)
    if num_sentences == 0
        return 0.0
    end
    sum = 0.0
    for sentence in sentences
        log_p_x = compute_log_forward_probability(trainer.model.sampler, sentence, true)
        sum += log_p_x
    end
    return sum
end

function compute_log_likelihood_train(trainer::Trainer)
    return compute_log_likelihood(trainer, trainer.dataset.train_sentences)
end

function compute_log_likelihood_dev(trainer::Trainer)
    return compute_log_likelihood(trainer, trainer.dataset.dev_sentences)
end

function print_segmentations(trainer::Trainer, num_to_print::Int, sentences::Vector{Sentence}, rand_indices::Vector{Int})
    num_to_print = min(length(sentences), num_to_print)
    for n in 1:num_to_print
        sentence_index = rand_indices[n]
        sentence = Sentence(sentences[sentence_index].sentence_string)
        # I don't think I fully understood why it's necessary to decode the sentence again if it's already segmented... Let's see.
        segment_lengths = viterbi_decode(trainer.model.sampler, sentence)
        split_sentence(sentence, segment_lengths)
        show(sentence)
        print("\n")
    end
end

function print_segmentations_train(trainer::Trainer, num_to_print::Int)
    return print_segmentations(trainer, num_to_print, trainer.dataset.train_sentences, trainer.rand_indices_train)
end

function print_segmentations_dev(trainer::Trainer, num_to_print::Int)
    shuffle!(trainer.rand_indices_dev)
    return print_segmentations(trainer, num_to_print, trainer.dataset.dev_sentences, trainer.dataset.rand_indices_dev)
end

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
    sentences = [[c for c in line if !Base.isspace(c)] for line in readlines(f) if !isempty(line)]
    # Here each entry in `sentences` is an array of Char. Need to convert it into an actual string.
    for char_array in sentences
        add_sentence(corpus, utf32(String(char_array)))
    end
end

export train
function train(corpus_path, output_path, split_proportion = 0.9, epochs = 100000, max_word_length = 4)
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
        # sample_hyperparameters(trainer)
        sample_lambda(trainer)

        # The accuracy is better after several iterations have been already done.
        if epoch > 3
            update_p_k_given_chpylm(trainer)
        end

        elapsed_time = time() - start_time
        println("Iteration $(epoch), Elapsed time in this iteration: $(elapsed_time)")
        if epoch % 10 == 0
            print_segmentations_train(trainer, 10)
            println("Perplexity_dev: $(compute_perplexity_dev(trainer))")
            sample_hyperparameters(trainer)
        end
    end

    out = open(output_path, "w")
    serialize(out, model)
    close(out)
end
end
