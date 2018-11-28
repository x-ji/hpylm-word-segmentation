include("Def.jl")
include("Corpus.jl")
include("Model.jl")
using OffsetArrays
# Actually I'm not sure if we really need such a complicated Trainer class. Let's first go on though.

mutable struct Trainer
    rand_indices_train::Vector{UInt}
    rand_indices_dev::Vector{UInt}
    dataset::Dataset
    vocabulary::Vocabulary
    model::Model
    chpylm_sampling_probability_table::Vector{Float64}
    chpylm_sampling_id_table::Vector{Char}
    always_accept_new_segmentation::Bool
    # What does this mean
    added_to_chpylm_train::Vector{Bool}
    num_segmentation_rejections::UInt
    num_segmentation_acceptances::UInt
    function Trainer(dataset::Dataset, model::Model, always_accept_new_segmentation::Bool)
        trainer = new()
        trainer.dataset = dataset
        trainer.model = model
        trainer.vocabulary = dataset.vocabulary
        # Need extra space for EOS?
        trainer.chpylm_sampling_probability_table = Vector{Float64}(undef, get_num_characters(trainer.vocabulary) + 1)
        trainer.chpylm_sampling_id_table = Vector{Char}(undef, get_num_characters(trainer.vocabulary) + 1)
        trainer.added_to_chpylm_train = fill(false, (length(dataset.train_sentences)))
        trainer.rand_indices_train = Vector{UInt}(undef, length(dataset.train_sentences))
        for i in 1:length(dataset.train_sentences)
            trainer.rand_indices_train[i] = i
        end
        trainer.rand_indices_dev = Vector{UInt}(undef, length(dataset.dev_sentences))
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
Sample lambda values for different types of characters
"""
function sample_lambda(trainer::Trainer)
    a_array = zeros(NUM_WORD_TYPES + 1, Float64)
    b_array = zeros(NUM_WORD_TYPES + 1, Float64)
    word_ids::Set{UInt} = Set()
    for t in 1:NUM_WORD_TYPES
        a_array[t] = trainer.npylm.λ_a
        b_array[t] = trainer.npylm.λ_b
    end
    for sentence in trainer.dataset.train_sentences
        # Go through each word in the sentence, excluding the BOS and EOS tokens.
        for index in 3:sentence.num_segments - 1
            word::String = get_nth_word_string(sentence, index)
            word_id::UInt = get_nth_word_id(sentence, index)
            word_length::UInt = get_nth_segment_length(sentence, index)
            if word_length > trainer.npylm.max_word_length
                continue
            end

            if !in(word_id, word_ids)
                tablegroups = trainer.npylm.whpylm.root.tablegroups[word_id]
                num_tablegroups = length(tablegroups)
                # TODO: Need to properly detect the word type.
                t = 1
                a_array[t] += num_tablegroups * word_length
                b_array[t] += num_tablegroups
                push!(word_ids, word_id)
            end
        end
        for t in 1:NUM_WORD_TYPES
            dist = Gamma(a_array[t], 1 / b_array[t])
            trainer.npylm.λ_for_types[t] = rand(dist, Float64)
        end
    end
end

"""
This function tries to generate a word randomly from the CHPYLM. Used by the function `update_p_k_given_chpylm`.

`skip_eow` means that EOW shouldn't be generated as the next char, because there is only BOW in the current word so far.
"""
function sample_next_char_from_chpylm_given_context(trainer::Trainer, context_chars::Vector{Char}, sample_t::UInt, skip_eow::Bool)
    context_length = length(context_chars)
    prob_sum = 0.0
    chpylm = trainer.model.npylm.chpylm
    table_index = 1
    all_characters = trainer.vocabulary.all_characters
    num_characters = length(all_characters)
    for c in all_characters
        p_w = compute_p_w_given_h(chpylm, c, context_chars, 1, context_length)
        prob_sum += p_w
        trainer.chpylm_sampling_probability_table[table_index] = p_w
        trainer.chpylm_sampling_id_table[table_index] = c
        table_index += 1
    end

    # Also record EOW as a probable character to be sampled.
    if !skip_eow
        p_w = compute_p_w_given_h(chpylm, EOW, context_chars, 1, context_length)
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
function update_p_k_given_chpylm(trainer::Trainer, num_samples::UInt = 20000, early_stopping_threshold::UInt = 10)
    max_word_length = get_max_word_length(trainer.model)
    p_k_chpylm = trainer.model.npylm.p_k_chpylm
    # Do you mean num_characters. Eh.
    # It's 1 longer than the original max_word_length, probably we have 0 in order to incorporate the possibility of getting length 0 word?
    # This array keeps track of total numbers of words of length k.
    # max_word_length + 1 because also a special case of k > max_word_length needs to be tracked?
    num_words_of_length_k = OffsetVector{UInt}(0, 0:max_word_length + 1)
    for i in 0:max_word_length + 1
        p_k_chpylm[i] = 0.0
    end

    wrapped_chars = Vector{Char}[max_word_length + 3]
    num_words_sampled = 0
    for itr in 1:num_samples
        wrapped_chars[1] = BOW

        # Keeps track of the actual word length
        cur_word_length = 0
        for j in 1:max_word_length
            # If we're only at the beginning of the sentence we shouldn't sample an EOW, because that would result in the "word" containing no characters at all.
            skip_eow = (j == 1) ? true : false
            next_char = sample_next_char_from_chpylm_given_context(trainer, wrapped_chars, j + 1, skip_eow)
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

        num_words_of_length_k[cur_word_length] += 1

        # If all possible lengths have enough data generated, we can terminate the sampling early.
        if itr % 100 == 0
            can_stop = true
            for k in 1:max_word_length + 1
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

    for k in 1:max_word_length + 1
        # Put in a Laplace smoothing over the final figures.
        p_k_chpylm[k] = (num_words_of_length_k[k] + 1) / (num_words_sampled + max_word_length + 1)
    end
end

function blocked_gibbs_sampling(trainer::Trainer)
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
            if trainer.added_to_chpylm_train[sentence_index]
                for n in 3:sentence.num_segments
                    remove_customer_at_index_n(trainer.model.npylm, sentence, n)
                end
            end
            # Because this is supervised data, i.e. guaranteed to be the true segmentation, we don't need to resample the sentence at all.
            for n in 3:sentence.num_segments
                add_customer_at_index_n(trainer.model.npylm, sentence, n)
            end
            trainer.added_to_chpylm_train[sentence_index] = true
        else
            old_segment_lengths = Vector{Int}(undef, max_sentence_length + 3)
            num_old_segments = 0
            old_log_p_s = 0.0
            new_log_p_s = 0.0

            # TODO: I thought this has more to do with the iteration of sampling? Do we really need such a mechanism anyways. But where is the iteration number in the first place eh.
            if trainer.added_to_chpylm_train[sentence_index]
                for n in 3:sentence.num_segments
                    remove_customer_at_index_n(trainer.model.npylm, sentence, n)
                end
            end

            # We need to later decide by some criteria whether to accept the new segmentation or just keep the old one.
            if trainer.always_accept_new_segment_lengths = false
                num_old_segments = get_num_segments_without_special_tokens(sentence)
                for i in 1:num_old_segments
                    # We save the old segmentation but get rid of the BOS and EOS tokens
                    # Two BOS in the beginning.
                    old_segment_lengths[i] = sentence.segments[i + 2]
                end
                old_log_p_s = compute_log_probability_of_sentence(trainer.model.npylm, sentence)
            end

            # Produce the new segmentation
            new_segment_lengths = blocked_gibbs_segment(trainer.model.sampler, sentence, true)
            split(sentence, new_segment_lengths)

            # TODO: There might be a way to avoid performing the check twice? Using a single Sentence struct to hold all these stuffs is quite a bit restrictive.
            if trainer.always_accept_new_segment_lengths = false
                old_log_p_s = compute_log_probability_of_sentence(trainer.model.npylm, sentence)
                # When the log probability of the new segmentation is lower, accept the new segmentation only with a certain probability
                bernoulli = min(1.0, exp(new_log_p_s - old_log_p_s))
                r = rand(Float64)
                if bernoulli < r
                    split(sentence, old_segment_lengths, num_old_segments)
                    trainer.num_segmentation_rejections += 1
                else
                    trainer.num_segmentation_acceptances += 1
                end
            end

            for n in 3:sentence.num_segments
                add_customer_at_index_n(trainer.model.npylm, sentence, n)
            end
            trainer.added_to_chpylm_train[sentence_index] = true
        end


    end
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
        split(sentence, segment_lengths)
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

function compute_log_likelihood(trainer::Trainer, sentences::Vector{Sentence})
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

function print_segmentations(trainer::Trainer, num_to_print::UInt, sentences::Vector{Sentence}, rand_indices::Vector{UInt})
    num_to_print = min(length(sentences), num_to_print)
    for n in 1:num_to_print
        sentence_index = rand_indices[n]
        sentence = Sentence(sentences[sentence_index].sentence_string)
        # I don't think I fully understood why it's necessary to decode the sentence again if it's already segmented... Let's see.
        segment_lengths = viterbi_decode(trainer.model.sampler, sentence)
        split(sentence, segment_lengths)
        show(sentence)
    end
end

function print_segmentations_train(trainer::Trainer, num_to_print::UInt)
    return print_segmentations(trainer, num_to_print, trainer.dataset.train_sentences)
end

function print_segmentations_dev(trainer::Trainer, num_to_print::UInt)
    shuffle!(trainer.rand_indices_dev)
    return print_segmentations(trainer, num_to_print, trainer.dataset.dev_sentences)
end