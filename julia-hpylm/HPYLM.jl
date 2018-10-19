__precompile__()
module HPYLM

include("Prior.jl")
include("PYP.jl")
include("Corpus.jl")
include("PYPLM.jl")

import Base.show

        end

        if it % 10 == 1
            println("Model: $model")
            ll = log_likelihood(model)
            perplexity = exp(-ll / (n_words + n_sentences))
            println("ll=$ll, ppl=$perplexity")
        end

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
    println("Reading training corpus")

    # We first construct the character corpus one character at a time.
    f = open(corpus_path)
    training_corpus = read_corpus(f, char_vocab)
    close(f)

    # +I think we should actually need a Poisson correction over this?+
    # See p.102 Section 3
    # Previously, the lexicon is finite, so we could just use a uniform prior. But here, because now the lexicon are all generated from the word segmentation, the lexicon becomes *countably infinite*.
    # Here is where we need to make modifications. The initial base, i.e. G_0, in the original Teh model is just a uniform distribution. But here we need to make it another distribution, a distribution which is based on a character-level HPYLM.
    # All distributions should have the same interface, have the same set of methods that enable sampling and all that. Therefore there must be some sort of "final form" of the character HPYLM, from which this word-level HPYLM can fall back upon.
    # We just need to define and initialize that distribution somewhere.

    # The final base measure for the character HPYLM, as described in p. 102 to the right of the page.
    # This should actually be "uniform over the possible characters" of the given language. IMO this seems to suggest importing a full character set for Chinese or something. But just basing it on the training material shouldn't hurt? Let's see then.
    character_base = UniformDist(length(char_vocab))
    character_model = PYPContainer(3, character_base)
    
    # TODO: Create a special type for character n-gram model and use that directly.
    # TODO: They used Poisson distribution to correct for word length (later).
    # This is the whole npylm. Its outmost layer is the word HPYLM, while the base of the word HPYLM is the char HPYLM.
    npylm = PYPContainer(3, character_model)

    # initial_base = UniformDist(length(char_vocab))
    # model = PYPContainer(order, initial_base)

    println("Training model")

    # Then it's pretty much the problem of running the sampler.
    blocked_gibbs_sampler(npylm, training_corpus, iter, 100)

    # Also useful when serializing
    # npylm.char_vocab = char_vocab
    # npylm.word_vocab = word_vocab
    out = open(output_path, "w")
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
function blocked_gibbs_sampler(npylm::PYPContainer, corpus::Array{Array{Int,1},1}, n_iter::Int, mh_iter::Int)
    # We need to store the segmented sentences somewhere, mostly because we need to remove the sentence data when the iteration count is >= 2.
    # I might need to change the type in which the sentences are stored. for now it's Array{Array{Int,1},1}, just as in the original vpyp program: Each word is represented as an Integer, and when we need to get its representation character-by-character, we need to go through some sort of conversion process.
    # Actually a two-dimensional array might be more sensible. But I can worry about optimizations later. The corpus is read in as a nested array because of the way list comprehension works.
    total_n_sentences = length(corpus)
    # What's the way to initialize an array of objects? This should work already by allocating the space, I guess.
    segmented_sentences::Array{Array{Int,1},1} = Array{Array{Int,1},1}(total_n_sentences)

    # This doesn't seem to be necessary
    # for i in 1:length(corpus)
    #     segmented_sentences[i] = []
    # end

    # Run the blocked Gibbs sampler for this many iterations.
    for it in 1:n_iter
        println("Iteration $it/$n_iter")

        # I think a more reasonable way is to simply shuffle an array of all the sentences, and make sure that in each iteration, every sentence is selected, albeit in a random order, of course.
        sentence_indices = shuffle(1:length(corpus))

        for index in sentence_indices
            # First remove the segmented sentence data from the NPYLM
            if it > 1
                # How do we actually do it though. Since we can't just uniformly generate fixed n-grams from this bag of characters nonchalently, it seems that we actually need to keep track of all the prevoius segmentations of the strings, if I understood it correctly.
                # Let me proceed with other parts of this function first and return to this a bit later, I guess.
                # OK. Done it. Great!
                remove_sentence_from_model(segmented_sentences[index], npylm)
            end
            # Get the raw, unsegmented sentence and segment it again.
            selected_sentence = corpus[index]
            segmented_sentence = sample_segmentation(selected_sentence)
            # Add the segmented sentence data to the NPYLM
            add_sentence_to_model(segmented_sentence, npylm)
            # Store the (freshly segmented) sentence so that we may remove its segmentation data in the next iteration.
            segmented_sentences[index] = segmented_sentence
        end

    end

    # In the paper (Figure 3) they seem to be sampling the hyperparameters at every iteration.
    println("Resampling hyperparameters")
    acceptance, rejection = resample_hyperparameters(npylm, mh_iter)
    acceptancerate = acceptance / (acceptance + rejection)
    println("MH acceptance rate: $acceptancerate")
    # println("Model: $model")
    # ll = log_likelihood(npylm)
    # perplexity = exp(-ll / (n_words + n_sentences))
    # println("ll=$ll, ppl=$perplexity")
end

# These two are helper functions to the whole blocked Gibbs sampler.
function add_sentence_to_model(npylm::PYPContainer, sentence::Array{Int,1})
    sentence_ngrams = ngrams(sentence, npylm.order)
    for ngram in sentence_ngrams
        increment(npylm, ngram[1:end - 1], ngram[end])
    end
end

function remove_sentence_from_model(npylm::PYPContainer, sentence::Array{Int,1})
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
function sample_segmentation(sentence::Array{Int,1}, max_word_length::Int, npylm::PYPContainer)::Array{Int,1}
    N = length(sentence)
    # Initialize to a negative value. If we see the value is negative, we know this box has not been filled yet.
    prob_matrix = fill(-1.0, (N, N))
    # First run the forward filtering
    for t in 1:N
        for k in max(1, t - max_word_length):t
            forward_filtering(sentence, t, k, prob_matrix, npylm)
        end
    end

    segmentation_output = []
    t = N
    i = 0
    # The sentence boundary symbol. Should it be the STOP symbol?
    # OK are those two even the same thing? The START, STOP defined in Corpus and the sentence boundary symbol. Guess I'll have to look at it a bit more then.
    w = STOP

    while t > 0
        # OK I think I get it.
        # It's a bit messy to implement a function just for this `draw` procedure here. Maybe let's just directly write out the procedures anyways.
        probabilities = zeros(Array{Float64,1}, max_word_length)
        # Keep the w and try out different variations of k, so that different segmentations serve as different context words to the w.
        for k in 1:max_word_length
            cur_segmentation = sentence[t - k + 1:t]
            # cur_context = charseq_to_string(cur_segmentation, global char_vocab, global word_vocab)
            cur_context = charseq_to_string(cur_segmentation)
            # cur_word = charseq_to_string(w, global char_vocab, global word_vocab)
            cur_word = charseq_to_string(w)
            probabilities[k] = prob(npylm, cur_context, cur_word);
        end

        # Draw value k from the weights calculated above.
        k = sample(1:max_word_length, Weights(probabilities))
        # This is now the newest word we sampled.
        # The word representation should be converted from the char representation then.
        sampled_word = charseq_to_string(sentence[(t - k + 1):t])

        # Update w which indicates the last segmented word.
        w = charseq_to_string(sampled_word)

        push!(segmentation_output, sampled_word)
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
function forward_filtering(sentence::Array{Int,1}, t::UInt, k::UInt, prob_matrix::Array{Float64,2}, npylm::PYPContainer)::Float64
    # Base case: α[0][n] = 1
    # Another way to do it is to just initialize the array as such. But let's just keep it like this for now, as this is more the responsibility of this algorithm?
    if (t == 0)
        return 1.0
    end
    if (prob_matrix[t][k] >= 0.0)
        return prob_matrix[t][k]
    end
    temp::Float64 = 0.0
    for j = 1:(t - k)
        # The probability here refers to the bigram probability of two adjacent words.
        # Therefore we likely need to convert this thing to word integers first.

        # It's really good that Julia's indexing system makes perfect sense and matches up with the mathematical notation used in the paper perfectly.
        # string_rep_potential_context = charseq_to_string(sentence[(t - k - j + 1):(t - k)], global char_vocab, global word_vocab)
        string_rep_potential_context = charseq_to_string(sentence[(t - k - j + 1):(t - k)])
        # string_rep_potential_word = charseq_to_string(sentence[(t - k + 1):t], global char_vocab, global word_vocab)
        string_rep_potential_word = charseq_to_string(sentence[(t - k + 1):t])
        bigram_prob = prob(npylm, string_rep_potential_context, string_rep_potential_word)

        temp += bigram_prob * forward_filtering(sentence, (t - k), j, prob_matrix, npylm)
    end

    # Store the final value in the DP matrix.
    prob_matrix[t][k] = temp
    return temp
end

"""
Helper function to forward_filtering

Convert a sequence of characters (represented in Int) to string (represented in Int)
"""
# function charseq_to_string(char_seq::Array{Int,1}, char_vocab::Vocabulary, word_vocab::Vocabulary)::Int
function charseq_to_string(char_seq::Array{Int,1})::Int
    # First: Convert the (int) character sequence back to their original coherent string
    # Wait, if those two are already global, I wouldn't need to pass them in as arguments anymore.
    string::String = join(map(char_int->get(global char_vocab, char_int), char_seq), "")
    # Then: Lookup the string in the word vocab
    string_rep::Int = get(global word_vocab, string)
    return string_rep
end


"""
Helper function to sample_segmentation

Function to proportionally draw a k given the prob matrix and indices.

p. 104, Figure 5, line 8

The real thing to do is to actually just draw the thing by:
- Fixing the "dish" (word)
- Changing the context by gradually increasing k
"""
# function draw(sentence::Array{Int, 1}, i::Int, t::Int, w::Int, max_word_length::Int, prob_matrix::Array{Float64,2}, npylm::PYPContainer)
#     probabilities = zeros(Array{Float64,1}, max_word_length)
#     # Keep the w and try out different variations of k, so that different segmentations serve as different context words to the w.
#     for k in 1:max_word_length
#         curr_segmentation = 
#     prob = prob()
#     end
# end

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
