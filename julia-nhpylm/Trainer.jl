include("Def.jl")
include("Corpus.jl")
include("Model.jl")
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
        trainer.rand_indices_dev= Vector{UInt}(undef, length(dataset.dev_sentences))
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

"Sample lambda values for different types of characters"
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

function sample_word_from_chpylm_given_context(trainer::Trainer, context_ids::Vector{Char}, sample_t::UInt, skip_eow::Bool)
    prob_sum = 0.0
    chpylm = trainer.model.npylm.chpylm
    table_index = 1
    all_characters = trainer.vocabulary.all_characters
    num_characters = length(all_characters)
    for c in all_characters
        p_w = compute_p_w
    end
end

function print_segmentation(num_to_print::UInt, sentences::Vector{Sentence}, rand_indices::Vector{UInt})
end