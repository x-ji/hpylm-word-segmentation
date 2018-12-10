__precompile__()
module HPYLM
using Distributions
using SpecialFunctions
using Serialization
using Random
using StatsBase

include("Prior.jl")
include("Corpus.jl")
include("UniformDist.jl")
include("PYP.jl")

import Base.show

export train
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
    println("Reading training corpus")

    f1 = open(corpus_path)
    char_vocab = Vocabulary()
    read_corpus(f1, char_vocab)
    close(f1)

    f2 = open(corpus_path)
    # Now let me just try to directly use the string without the Vocab to see the performance tradeoffs.
    training_corpus = [[string(c) for c in line if !Base.isspace(c)] for line in readlines(f2) if !isempty(line)]
    close(f2)

    # See p.102 Section 3
    # Previously, the lexicon is finite, so we could just use a uniform prior. But here, because now the lexicon are all generated from the word segmentation, the lexicon becomes *countably infinite*.
    # Here is where we need to make modifications. The initial base, i.e. G_0, in the original Teh model is just a uniform distribution. But here we need to make it another distribution, a distribution which is based on a character-level HPYLM.
    # All distributions should have the same interface, have the same set of methods that enable sampling and all that. Therefore there must be some sort of "final form" of the character HPYLM, from which this word-level HPYLM can fall back upon.
    # We just need to define and initialize that distribution somewhere.

    # The final base measure for the character HPYLM, as described in p. 102 to the right of the page.
    # This should actually be "uniform over the possible characters" of the given language. IMO this seems to suggest importing a full character set for Chinese or something. But just basing it on the training material first shouldn't hurt? Let's see then.

    # ... Good question, how do I know the size of the vocabulary now that I don't actually construct a char struct?
    # Guess I'll still have to make use of the char_vocab here, once and for all then.
    character_base = UniformDist(length(char_vocab))
    # False means this is for chars, not words.
    # character_model = PYPContainer(3, character_base, false)
    # What if I simply use a bigram model first. Damn it.
    character_model = PYPContainer(2, character_base, false)

    # TODO: Create a special type for character n-gram model and use that directly.
    # TODO: They used Poisson distribution to correct for word length (later).
    # This is the word npylm container, while the base of the word HPYLM is the char HPYLM.
    npylm = PYPContainer(2, character_model, true)

    println("Training model")

    # Then it's pretty much the problem of running the sampler.
    blocked_gibbs_sampler(npylm, training_corpus, iter, 100)

    # Also useful when serializing
    out = open(output_path, "w")
    # total_model = Model(npylm, char_vocab, word_vocab)
    # serialize(out, total_model)
    serialize(out, npylm)
    close(out)
end

"""
The blocked Gibbs sampler. Function that arranges the entire sampling process for the training.

The previous approach is to just use a Gibbs sampler to randomly select a character and draw a simple binary decision about whether there's a word boundary there. Each such decision would trigger an update of the language model.
  - That was slow and would not converge without annealing. Required 20000 sampling for each character in the training data.
  - Only works on a bigram level.

Here we use a blocked Gibbs sampler:
1. A sentence is *randomly selected* (out of all the sentences in the training data).
2. Remove the unit ("sentence") data of its word segmentation from the NPYLM (it should be cascaded between the two PYLMs anyways).
  Though I suppose the "removal" can only happen starting from the second iteration, as usual.
3. Sample a new segmentation on this sentence using our sampling algorithm.
4. Add the sampled unit ("sentence") data back to the NPYLM based on this new segmentation.
"""
function blocked_gibbs_sampler(npylm::PYPContainer, corpus::Array{Array{String,1},1}, n_iter::Int, mh_iter::Int, print_segmented_sentences::Bool=false)
    # We need to store the segmented sentences somewhere, mostly because we need to remove the sentence data when the iteration count is >= 2.
    # TODO: I might need to change the type in which the sentences are stored. For now it's Array{Array{String,1},1}, the corpus is read in as a nested array because of the way list comprehension works. Actually a two-dimensional array might be more sensible. But I can worry about optimizations later.
    total_n_sentences = length(corpus)
    segmented_sentences::Array{Array{String,1},1} = fill(String[], total_n_sentences)

    # Run the blocked Gibbs sampler for this many iterations.
    for it in 1:n_iter
        println("Iteration $it/$n_iter")

        # Simply shuffle an array of all the sentences, and make sure that in each iteration, every sentence is selected, albeit in a random order.
        sentence_indices = shuffle(1:length(corpus))

        for index in sentence_indices
            # First remove the segmented sentence data from the NPYLM
            if it > 1
                remove_sentence_from_model(npylm, segmented_sentences[index])
            end
            # Get the raw, unsegmented sentence and segment it again.
            selected_sentence = corpus[index]
            # According to the paper they put the max_word_length at 4.
            segmented_sentence = sample_segmentation(selected_sentence, 4, npylm)
            # Add the segmented sentence data to the NPYLM
            add_sentence_to_model(npylm, segmented_sentence)
            # Store the (freshly segmented) sentence so that we may remove its segmentation data in the next iteration.
            segmented_sentences[index] = segmented_sentence
        end

        # We just try to show what the segmentation results look like in terms of actual strings.
        if print_segmented_sentences
            for sentence in segmented_sentences
                # Now, each entry in the "sentence" should be a proper word already.
                for word in sentence
                    print(word)
                    print(" ")
                end
                print("\n")
            end
        end

        if it % 5 == 0
            # TODO: In the paper (Figure 3) they seem to be sampling the hyperparameters at every iteration. We may choose to do it a bit less frequently.
            println("Resampling hyperparameters")
            acceptance, rejection = resample_hyperparameters(npylm, mh_iter)
            acceptancerate = acceptance / (acceptance + rejection)
            println("MH acceptance rate: $acceptancerate")
            # println("Model: $model")
            # ll = log_likelihood(npylm)
            # perplexity = exp(-ll / (n_words + n_sentences))
            # println("ll=$ll, ppl=$perplexity")
        end

    end

end

# These two are helper functions to the whole blocked Gibbs sampler.
function add_sentence_to_model(npylm::PYPContainer, sentence::Array{String,1})
    sentence_ngrams = ngrams(sentence, npylm.order)
    for ngram in sentence_ngrams
        increment(npylm, ngram[1:end - 1], ngram[end])
    end
end

function remove_sentence_from_model(npylm::PYPContainer, sentence::Array{String,1})
    sentence_ngrams = ngrams(sentence, npylm.order)
    for ngram in sentence_ngrams
        decrement(npylm, ngram[1:end - 1], ngram[end])
    end
end

"""
Function to the run the forward-backward inference which samples a sentence segmentation.

Sample a segmentation **w** for each string *s*.

p. 104, Figure 5
"""
function sample_segmentation(sentence::Array{String,1}, max_word_length::Int, npylm::PYPContainer)::Array{String,1}
    N = length(sentence)
    # Initialize to a negative value. If we see the value is negative, we know this box has not been filled yet.
    prob_matrix = fill(-1.0, (N, N))
    # First run the forward filtering
    for t in 1:N
        # The candidate word will actually be (t - k + 1):t. Therefore, By making sure that k cannot be smaller than t - max_word_length, we're ensuring that the candidate word will not be longer than:
        # t - (t - k + 1) + 1 = k
        # Wait so this still isn't right. What the hell. k can still be up to t long???
        for k in max(1, t - max_word_length):t
        # for k in 1:min(max_word_length, t)
            # Maybe this should be sentence[1:t] then.
            forward_filtering(sentence, t, k, prob_matrix, npylm)
        end
    end

    segmentation_output = []
    t = N
    i = 0
    # The sentence boundary symbol. Should it be the STOP symbol?
    # TODO: Then we might have no need for any START symbol after all. May need to revise the code.
    w = STOP

    while t > 0
        # OK I think I get it.
        # It's a bit messy to implement a function just for this `draw` procedure here, as too many arguments are involved. Maybe let's just directly write out the procedures anyways.
        # TODO: Turn it into a function.
        probabilities = fill(0.0, max_word_length)
        # The idea: Keep the w and try out different variations of k, so that different segmentations serve as different context words to the w.
        # Seems that sometimes the max_word_length could be just too big.
        # TODO: OK apparently I manually added this restriction for max_word_length while it just simply isn't there in the original paper. I wonder if something is getting mixed up in the process. Must figure this out.
        for k in 1:min(max_word_length, t)
            cur_segmentation = sentence[t - k + 1:t]
            cur_context = Base.join(cur_segmentation)
            # Need to convert this thing to an array, even though it's just the bigram case. In the trigram case there should be two words instead of one.
            # TODO: In trigram case we need to do something different.
            # Memory cost > 100MB. Maybe it's because we're trying to construct a context array on the fly? This is a bit weird.
            # Can definitely ask for advice over this. Or try some different array construction methods first etc. then.
            probabilities[k] = prob(npylm, [cur_context], w)
        end

        # Draw value k from the weights calculated above.
        k = sample(1:max_word_length, Weights(probabilities))

        # This is now the newest word we sampled.
        # The word representation should be converted from the char representation then.
        # Update w which indicates the last segmented word.
        w = Base.join(sentence[(t - k + 1):t])
        push!(segmentation_output, w)
        t = t - k
        i += 1
    end

    # The segmented outputs should be properly output in a reverse order.
    return reverse(segmentation_output)
end

"""
Helper function to sample_segmentation. Forward filtering is a part of the algorithm (line 3)

Algorithm documented in section 4.2 of the Mochihashi paper. Equation (7)
Compute α[t][k]
"""
function forward_filtering(sentence::Array{String,1}, t::Int, k::Int, prob_matrix::Array{Float64,2}, npylm::PYPContainer)::Float64
    # Base case: α[0][n] = 1
    if (t == 0)
        return 1.0
    end
    if (prob_matrix[t,k] >= 0.0)
        return prob_matrix[t,k]
    end
    temp::Float64 = 0.0
    for j = 1:(t - k)
        # The probability here refers to the bigram probability of two adjacent words.
        # Therefore we likely need to convert this thing to word integers first.

        # It's really good that Julia's 1-based indexing system makes perfect sense and matches up with the mathematical notation used in the paper perfectly.
        string_rep_potential_context = Base.join(sentence[(t - k - j + 1):(t - k)])
        string_rep_potential_word = Base.join(sentence[(t - k + 1):t])
        # Memory cost: > 600MB. Not sure maybe it also has something to do with the fact that we're creating an array on the fly? Is there any better way though?
        println("Potential context: $string_rep_potential_context, potential word: $string_rep_potential_word")
        bigram_prob = prob(npylm, [string_rep_potential_context], string_rep_potential_word)

        temp += bigram_prob * forward_filtering(sentence, (t - k), j, prob_matrix, npylm)
    end

    # Store the final value in the DP matrix.
    prob_matrix[t,k] = temp
    return temp
end

# function print_ppl(model::PYPContainer, corpus::Array{Array{Int,1},1})
#     n_sentences = length(corpus)
#     n_words = sum(length, corpus)
#     processed_corpus = map(sentence->ngrams(sentence, model.order), corpus)
#     n_oovs = 0
#     ll = 0.0

#     for sentence in processed_corpus
#         for ngram in sentence
#             p = prob(model, ngram[1:end - 1], ngram[end])
#             if p == 0
#                 n_oovs += 1
#             else
#                 ll += log(p)
#         end
#         end
#     end
#     ppl = exp(-ll / (n_sentences + n_words - n_oovs))
#     # ppl = exp(-ll / (n_words - n_oovs))
#     println("Sentences: $n_sentences, Words: $n_words, OOVs: $n_oovs")
#     println("LL: $ll, perplexity: $ppl")
# end

"""
Shows the sampled segmentation on the test corpus, using an already trained model.

OK so eventually it still does seem that we shouldn't really put the vocabs as "global variables". It would cause some quite inconvenient issues, especially when I try to run the evaluation later.

Just make them into arguments to each function that needs them, I guess. Could be a bit inconvenient but generally speaking this should be the right thing to do for sure. Let's try it then.
"""
function test_segmentation(npylm::PYPContainer, corpus::Array{Array{String, 1}, 1})
    # Well I don't even think there's much of a difference between this process and the original training process right? We just run the Blocked Gibbs Sampler again on the test data and see what results are output, don't we?

    # Just run one go because we're not training, just trying to see the segmentation results immediately.
    println("Before blocked_gibbs_sampler")
    blocked_gibbs_sampler(npylm, corpus, 1, 100, true)
end

"""
Tries to generate text from the vocabulary and model learned from the training corpus.

This is currently not really the focus of the project though. Let's see.

There probably needs to be a sample function? But how.
First we'd need to define what is "generation". Is it based on words or is it based on characters?
Well if we indeed have a character n-gram model then I do think we can generate characters one at a time. There's no inherent problem with that.
Yeah I a kind of get it: This will be more or less a *reverse* process from the top-down segmentation process.
1. Generate a word from the character n-gram model. Be it a 3-gram or an infinite-gram or 2-gram or whatnot (with 2-gram, we will only need to keep chugging on using the last generated character as the "context" and generate the next character. If that next character turns out to be STOP, the word is generated. The spirit is similar with the 3-gram and essentially llthe infinite-gram as well.)
2. After one word is generated (i.e. the "STOP" symbol is reached), use this word itself as the context to sampl the next word from the *word-level HPYLM*.
  OK but then where does the character NPYLM come into play in those later words?
  Or did I just understand it wrong?? Maybe what I should do is to actually start with the word model from the beginning anyways. Shouldn't the word model already be using the knowledge from the character model implicitly when it calcualtes the n-gram probability, because the character model is a base model for the G_0 case?
  Interesting. I think yeah that might be closer to the spirit but unfortunately this language model is quite incapable of coming up with previously unseen words apparently.
  But yeah OK I think it's always true that you can't really


"""
# function generate_text(npylm::PYPContainer, num_of_sentences::Int)
#     for sentence_index in 1:num_of_sentences
#         prob()
#     end
# end

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
# I'll need to rewrite this method to fit the current structure of the project.
function evaluate(corpus_path, model_path)
    m_in = open(model_path)
    npylm = deserialize(m_in)
    close(m_in)

    println("Deserialization complete")

    c_in = open(corpus_path)
    # The model should also read in the previously unseen characters in the evaluation corpus properly.
    # evaluation_corpus = read_corpus(c_in, char_vocab)
    evaluation_corpus = [[string(c) for c in line if !Base.isspace(c)] for line in readlines(c_in) if !isempty(line)]

    close(c_in)

    println("Before test_segmentation")
    test_segmentation(npylm, evaluation_corpus)
end

end
