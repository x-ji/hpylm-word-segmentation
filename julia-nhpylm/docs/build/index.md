
<a id='Julia-NHPYLM-Documentation-1'></a>

# Julia-NHPYLM Documentation




<a id='Corpus.jl-1'></a>

## Corpus.jl

<a id='NHPYLM.Corpus' href='#NHPYLM.Corpus'>#</a>
**`NHPYLM.Corpus`** &mdash; *Type*.



This struct keeps track of all sentences from the corpus files, and optionally the "true" segmentations, if any.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L27-L29' class='documenter-source'>source</a><br>

<a id='NHPYLM.Dataset' href='#NHPYLM.Dataset'>#</a>
**`NHPYLM.Dataset`** &mdash; *Type*.



This struct holds all the structs related to a session/task, including the vocabulary, the corpus and the sentences produced from the corpus.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L65-L67' class='documenter-source'>source</a><br>

<a id='NHPYLM.Vocabulary' href='#NHPYLM.Vocabulary'>#</a>
**`NHPYLM.Vocabulary`** &mdash; *Type*.



This struct keeps track of all the characters in the target corpus.

This is necessary because in the character CHPYLM, the G_0 needs to be calculated via a uniform distribution over all possible characters of the target language.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L5-L9' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_sentence-Tuple{NHPYLM.Corpus,LegacyStrings.UTF32String}' href='#NHPYLM.add_sentence-Tuple{NHPYLM.Corpus,LegacyStrings.UTF32String}'>#</a>
**`NHPYLM.add_sentence`** &mdash; *Method*.



Add an individual sentence to the corpus


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L39-L41' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_sentence-Tuple{NHPYLM.Dataset,LegacyStrings.UTF32String,Array{NHPYLM.Sentence,1}}' href='#NHPYLM.add_sentence-Tuple{NHPYLM.Dataset,LegacyStrings.UTF32String,Array{NHPYLM.Sentence,1}}'>#</a>
**`NHPYLM.add_sentence`** &mdash; *Method*.



Add a sentence to the train or dev sentence vector of the dataset


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L155' class='documenter-source'>source</a><br>

<a id='NHPYLM.read_corpus-Tuple{NHPYLM.Corpus,IOStream}' href='#NHPYLM.read_corpus-Tuple{NHPYLM.Corpus,IOStream}'>#</a>
**`NHPYLM.read_corpus`** &mdash; *Method*.



Read the corpus from an input stream


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Corpus.jl#L46-L48' class='documenter-source'>source</a><br>


<a id='CType.jl-1'></a>

## CType.jl

<a id='NHPYLM.detect_ctype-Tuple{Char}' href='#NHPYLM.detect_ctype-Tuple{Char}'>#</a>
**`NHPYLM.detect_ctype`** &mdash; *Method*.



Detect the type of a given character


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CType.jl#L283-L285' class='documenter-source'>source</a><br>


<a id='WType.jl-1'></a>

## WType.jl

<a id='NHPYLM.detect_word_type-Tuple{LegacyStrings.UTF32String}' href='#NHPYLM.detect_word_type-Tuple{LegacyStrings.UTF32String}'>#</a>
**`NHPYLM.detect_word_type`** &mdash; *Method*.



Detect the type of a word. Useful when we need to estimate different lambda values for different word types.

Basically it uses a heuristics to see the proportion of character types in this word to assign a word type eventually.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/WType.jl#L138-L142' class='documenter-source'>source</a><br>

<a id='NHPYLM.is_dash-Tuple{Char}' href='#NHPYLM.is_dash-Tuple{Char}'>#</a>
**`NHPYLM.is_dash`** &mdash; *Method*.



This refers to the full-width dash used to mark long vowels in Hiragana/Katakana


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/WType.jl#L19' class='documenter-source'>source</a><br>


<a id='Sentence.jl-1'></a>

## Sentence.jl

<a id='NHPYLM.Sentence' href='#NHPYLM.Sentence'>#</a>
**`NHPYLM.Sentence`** &mdash; *Type*.



This struct holds everything that represents a sentence, including the raw string that constitute the sentence, and the potential segmentation of the sentence, if it is segmented, either via a preexisting segmentation or via running the forward-filtering-backward-sampling segmentation algorithm.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sentence.jl#L5-L7' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_substr_word_id-Tuple{NHPYLM.Sentence,Int64,Int64}' href='#NHPYLM.get_substr_word_id-Tuple{NHPYLM.Sentence,Int64,Int64}'>#</a>
**`NHPYLM.get_substr_word_id`** &mdash; *Method*.



Get the word id of the substring with start*index and end*index. Note that in Julia the end_index is inclusive.

Note that the `hash` method returns UInt! This makes sense because a 2-fold increase in potential hash values can actually help a lot.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sentence.jl#L105-L109' class='documenter-source'>source</a><br>


<a id='PYP.jl-1'></a>

## PYP.jl

<a id='NHPYLM.PYP' href='#NHPYLM.PYP'>#</a>
**`NHPYLM.PYP`** &mdash; *Type*.



Each node is essentially a Pitman-Yor process in the hierarchical Pitman-Yor language model

We use a type parameter because it can be either for characters (Char) or for words (UTF32String/UInt)

The root PYP (depth 0) contains zero context. The deeper the depth, the longer the context.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L19-L25' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_customer-Union{Tuple{T}, Tuple{PYP{T},T,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Bool,IntContainer}} where T' href='#NHPYLM.add_customer-Union{Tuple{T}, Tuple{PYP{T},T,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Bool,IntContainer}} where T'>#</a>
**`NHPYLM.add_customer`** &mdash; *Method*.



Adds a customer eating a certain dish to a node.

Note that this method is applicable to both the WHPYLM and the CHPYLM, thus the type parameter.

d*array and θ*array contain the d values and θ values for each depth of the relevant HPYLM (Recall that those values are the same for one single depth.)


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L243-L249' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_customer_to_table-Union{Tuple{T}, Tuple{PYP{T},T,Int64,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,IntContainer}} where T' href='#NHPYLM.add_customer_to_table-Union{Tuple{T}, Tuple{PYP{T},T,Int64,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,IntContainer}} where T'>#</a>
**`NHPYLM.add_customer_to_table`** &mdash; *Method*.



The second item returned in the tuple is the index of the table to which the customer is added.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L167' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_p_w_with_parent_p_w-Union{Tuple{T}, Tuple{PYP{T},T,Float64,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray}} where T' href='#NHPYLM.compute_p_w_with_parent_p_w-Union{Tuple{T}, Tuple{PYP{T},T,Float64,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray}} where T'>#</a>
**`NHPYLM.compute_p_w_with_parent_p_w`** &mdash; *Method*.



Compute the possibility of the word/char `dish` being generated from this pyp (i.e. having this pyp as its context). The equation is the one recorded in the original Teh 2006 paper.

When is*parent*p*w == True, the third argument is the parent*p*w. Otherwise it's simply the G*0.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L415-L419' class='documenter-source'>source</a><br>

<a id='NHPYLM.find_child_pyp-Union{Tuple{T}, Tuple{PYP{T},T}, Tuple{PYP{T},T,Bool}} where T' href='#NHPYLM.find_child_pyp-Union{Tuple{T}, Tuple{PYP{T},T}, Tuple{PYP{T},T,Bool}} where T'>#</a>
**`NHPYLM.find_child_pyp`** &mdash; *Method*.



Find the child PYP whose context is the given dish


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L149-L151' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_all_pyps_at_depth-Union{Tuple{T}, Tuple{PYP{T},Int64,OffsetArray{PYP{T},1,AA} where AA<:AbstractArray}} where T' href='#NHPYLM.get_all_pyps_at_depth-Union{Tuple{T}, Tuple{PYP{T},Int64,OffsetArray{PYP{T},1,AA} where AA<:AbstractArray}} where T'>#</a>
**`NHPYLM.get_all_pyps_at_depth`** &mdash; *Method*.



If run successfully, this function should put all pyps at the specified depth into the accumulator vector.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L586' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_max_depth-Union{Tuple{T}, Tuple{PYP{T},Int64}} where T' href='#NHPYLM.get_max_depth-Union{Tuple{T}, Tuple{PYP{T},Int64}} where T'>#</a>
**`NHPYLM.get_max_depth`** &mdash; *Method*.



Basically a DFS to get the maximum depth of the tree with this `pyp` as its root


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L522' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_num_nodes-Union{Tuple{PYP{T}}, Tuple{T}} where T' href='#NHPYLM.get_num_nodes-Union{Tuple{PYP{T}}, Tuple{T}} where T'>#</a>
**`NHPYLM.get_num_nodes`** &mdash; *Method*.



A DFS to get the total number of nodes with this `pyp` as the root


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L534' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_num_tables_serving_dish-Union{Tuple{T}, Tuple{PYP{T},T}} where T' href='#NHPYLM.get_num_tables_serving_dish-Union{Tuple{T}, Tuple{PYP{T},T}} where T'>#</a>
**`NHPYLM.get_num_tables_serving_dish`** &mdash; *Method*.



This function explicitly returns the number of **tables** (i.e. not customers) serving a dish!


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L132-L134' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_log_x_u-Union{Tuple{T}, Tuple{PYP{T},Float64}} where T' href='#NHPYLM.sample_log_x_u-Union{Tuple{T}, Tuple{PYP{T},Float64}} where T'>#</a>
**`NHPYLM.sample_log_x_u`** &mdash; *Method*.



Note that only the log of x_u is used in the final sampling, expression (41) of the Teh technical report. Therefore our function also only ever calculates the log. Should be easily refactorable though.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L607-L609' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_summed_y_ui-Union{Tuple{T}, Tuple{PYP{T},Float64,Float64}, Tuple{PYP{T},Float64,Float64,Bool}} where T' href='#NHPYLM.sample_summed_y_ui-Union{Tuple{T}, Tuple{PYP{T},Float64,Float64}, Tuple{PYP{T},Float64,Float64,Bool}} where T'>#</a>
**`NHPYLM.sample_summed_y_ui`** &mdash; *Method*.



Note that in expressions (40) and (41) of the technical report, the yui values are only used when they're summed. So we do the same here.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/PYP.jl#L620-L622' class='documenter-source'>source</a><br>


<a id='HPYLM.jl-1'></a>

## HPYLM.jl

<a id='NHPYLM.CHPYLM' href='#NHPYLM.CHPYLM'>#</a>
**`NHPYLM.CHPYLM`** &mdash; *Type*.



Character Hierarchical Pitman-Yor Language Model 

In this case the HPYLM for characters is an infinite Markov model, different from that used for the words.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L5-L9' class='documenter-source'>source</a><br>

<a id='NHPYLM.Model' href='#NHPYLM.Model'>#</a>
**`NHPYLM.Model`** &mdash; *Type*.



This is the struct that will serve as a container for the whole NHPYLM. it will be serialized after training.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L18-L20' class='documenter-source'>source</a><br>

<a id='NHPYLM.Trainer' href='#NHPYLM.Trainer'>#</a>
**`NHPYLM.Trainer`** &mdash; *Type*.



This struct contains everything needed for the training process


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L108' class='documenter-source'>source</a><br>

<a id='NHPYLM.WHPYLM' href='#NHPYLM.WHPYLM'>#</a>
**`NHPYLM.WHPYLM`** &mdash; *Type*.



Hierarchical Pitman-Yor process for words


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/WHPYLM.jl#L4-L6' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_customer_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}' href='#NHPYLM.add_customer_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}'>#</a>
**`NHPYLM.add_customer_at_index_n`** &mdash; *Method*.



This function adds the customer. See documentation above.

This is a version to be called from the NPYLM.

If the parent*p*w*cache is already set, then update the path*nodes as well.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L107-L113' class='documenter-source'>source</a><br>

<a id='NHPYLM.add_customer_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64}' href='#NHPYLM.add_customer_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64}'>#</a>
**`NHPYLM.add_customer_at_index_n`** &mdash; *Method*.



The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you

  * first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass*count or stop*count along the path of the tree
  * sample a new order (depth) according to the conditional probability
  * add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass*count or stop*count along the (new) path

This function adds the customer


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L92-L99' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_log_forward_probability-Tuple{NHPYLM.Model,LegacyStrings.UTF32String,Bool}' href='#NHPYLM.compute_log_forward_probability-Tuple{NHPYLM.Model,LegacyStrings.UTF32String,Bool}'>#</a>
**`NHPYLM.compute_log_forward_probability`** &mdash; *Method*.



Compute the log forward probability of any sentence given the whole NHPYLM model


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L99' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_log_p_w-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}' href='#NHPYLM.compute_log_p_w-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}'>#</a>
**`NHPYLM.compute_log_p_w`** &mdash; *Method*.



Compute the *log* probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L279-L281' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_p_w-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}' href='#NHPYLM.compute_p_w-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}'>#</a>
**`NHPYLM.compute_p_w`** &mdash; *Method*.



Compute the probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L271-L273' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_perplexity-Tuple{NHPYLM.Trainer,Array{NHPYLM.Sentence,1}}' href='#NHPYLM.compute_perplexity-Tuple{NHPYLM.Trainer,Array{NHPYLM.Sentence,1}}'>#</a>
**`NHPYLM.compute_perplexity`** &mdash; *Method*.



Compute the perplexity based on optimal segmentation produced by the Viterbi algorithm


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L386' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_num_nodes-Union{Tuple{HPYLM}, Tuple{T}} where T' href='#NHPYLM.get_num_nodes-Union{Tuple{HPYLM}, Tuple{T}} where T'>#</a>
**`NHPYLM.get_num_nodes`** &mdash; *Method*.



Get the total number of nodes in this HPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/HPYLM.jl#L18' class='documenter-source'>source</a><br>

<a id='NHPYLM.init_hyperparameters_at_depth_if_needed-Tuple{NHPYLM.HPYLM,Int64}' href='#NHPYLM.init_hyperparameters_at_depth_if_needed-Tuple{NHPYLM.HPYLM,Int64}'>#</a>
**`NHPYLM.init_hyperparameters_at_depth_if_needed`** &mdash; *Method*.



Sometimes the hyperparameter array can be shorter than the actual depth of the HPYLM (especially for CHPYLM whose depth is dynamic. In this case initialize the hyperparameters at the deeper depth.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/HPYLM.jl#L41' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_depth_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}' href='#NHPYLM.sample_depth_at_index_n-Tuple{NHPYLM.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}'>#</a>
**`NHPYLM.sample_depth_at_index_n`** &mdash; *Method*.



Sample the depth of the character at index n of the given characters (word).


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/CHPYLM.jl#L351-L353' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_hyperparameters-Tuple{NHPYLM.HPYLM}' href='#NHPYLM.sample_hyperparameters-Tuple{NHPYLM.HPYLM}'>#</a>
**`NHPYLM.sample_hyperparameters`** &mdash; *Method*.



Sample all hyperparameters of this HPYLM


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/HPYLM.jl#L102' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_lambda-Tuple{NHPYLM.Trainer}' href='#NHPYLM.sample_lambda-Tuple{NHPYLM.Trainer}'>#</a>
**`NHPYLM.sample_lambda`** &mdash; *Method*.



Sample lambda values for different types of characters.

For example, puncutation marks, alphabets, Chinese ideographs are all different types of characters.

Each type would get its own average word length correction with a different lambda value.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L152-L158' class='documenter-source'>source</a><br>

<a id='NHPYLM.sample_next_char_from_chpylm_given_context-Tuple{NHPYLM.Trainer,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,Bool}' href='#NHPYLM.sample_next_char_from_chpylm_given_context-Tuple{NHPYLM.Trainer,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,Bool}'>#</a>
**`NHPYLM.sample_next_char_from_chpylm_given_context`** &mdash; *Method*.



This function tries to generate a word randomly from the CHPYLM. Used by the function `update_p_k_given_chpylm`.

`skip_eow` means that EOW shouldn't be generated as the next char. This applies when there is only BOW in the current word so far.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L198-L202' class='documenter-source'>source</a><br>

<a id='NHPYLM.sum_auxiliary_variables_recursively-Union{Tuple{T}, Tuple{HPYLM,PYP{T},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Int64}} where T' href='#NHPYLM.sum_auxiliary_variables_recursively-Union{Tuple{T}, Tuple{HPYLM,PYP{T},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Int64}} where T'>#</a>
**`NHPYLM.sum_auxiliary_variables_recursively`** &mdash; *Method*.



Sum up all values of a auxiliary variable on the same depth into one variable.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/HPYLM.jl#L78' class='documenter-source'>source</a><br>

<a id='NHPYLM.update_p_k_given_chpylm' href='#NHPYLM.update_p_k_given_chpylm'>#</a>
**`NHPYLM.update_p_k_given_chpylm`** &mdash; *Function*.



This function updates the cache of the probability of sampling a word of length k from the CHPYLM.

As mentioned in Section 4.3 of the paper, a Monte Carlo method is employed to generate words randomly from the CHPYLM so that empirical estimates of p(k|chpylm) can be obtained.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/NHPYLM.jl#L230-L234' class='documenter-source'>source</a><br>


<a id='Sampler.jl-1'></a>

## Sampler.jl

<a id='NHPYLM.Sampler' href='#NHPYLM.Sampler'>#</a>
**`NHPYLM.Sampler`** &mdash; *Type*.



This structs holds all the necessary fields for sampling sentence segmentations using forward-backward inference.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L5-L7' class='documenter-source'>source</a><br>

<a id='NHPYLM.backward_sample_k_and_j-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64,NHPYLM.IntContainer,NHPYLM.IntContainer}' href='#NHPYLM.backward_sample_k_and_j-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64,NHPYLM.IntContainer,NHPYLM.IntContainer}'>#</a>
**`NHPYLM.backward_sample_k_and_j`** &mdash; *Method*.



Returns k and j in a tuple, which denote the offsets for word boundaries of the two words we are interested in sampling.

"next*word" really means the target word, the last gram in the 3 gram, e.g. the EOS in p(EOS | c^N*{N-k} c^{N-k}_{N-k-j})


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L289-L293' class='documenter-source'>source</a><br>

<a id='NHPYLM.backward_sampling-Tuple{NHPYLM.Sampler,NHPYLM.Sentence}' href='#NHPYLM.backward_sampling-Tuple{NHPYLM.Sampler,NHPYLM.Sentence}'>#</a>
**`NHPYLM.backward_sampling`** &mdash; *Method*.



Performs the backward sampling on the target sentence.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L231-L233' class='documenter-source'>source</a><br>

<a id='NHPYLM.blocked_gibbs_segment-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}' href='#NHPYLM.blocked_gibbs_segment-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}'>#</a>
**`NHPYLM.blocked_gibbs_segment`** &mdash; *Method*.



Does the segment part in the blocked Gibbs algorithm (line 6 of Figure 3 of the paper)


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L409' class='documenter-source'>source</a><br>

<a id='NHPYLM.compute_log_forward_probability-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}' href='#NHPYLM.compute_log_forward_probability-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}'>#</a>
**`NHPYLM.compute_log_forward_probability`** &mdash; *Method*.



Computes the probability of resulting in EOS with the given α_tensor for the sentence.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L636' class='documenter-source'>source</a><br>

<a id='NHPYLM.forward_filtering-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}' href='#NHPYLM.forward_filtering-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Bool}'>#</a>
**`NHPYLM.forward_filtering`** &mdash; *Method*.



Performs the forward filtering on the target sentence.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L117-L119' class='documenter-source'>source</a><br>

<a id='NHPYLM.get_substring_word_id_at_t_k-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64}' href='#NHPYLM.get_substring_word_id_at_t_k-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64}'>#</a>
**`NHPYLM.get_substring_word_id_at_t_k`** &mdash; *Method*.



α[t][k][j] represents the marginal probability of string c1...ct with both the final k characters and further j preceding characters being words.

This function returns the id of the word constituted by the last k characters of the total t characters.

Note that since this function already takes care of the index shift that's needed in Julia, the callers will still just call it normally.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L98-L104' class='documenter-source'>source</a><br>

<a id='NHPYLM.viterbi_argmax_backward_sample_k_and_j_to_eos-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64,NHPYLM.IntContainer,NHPYLM.IntContainer}' href='#NHPYLM.viterbi_argmax_backward_sample_k_and_j_to_eos-Tuple{NHPYLM.Sampler,NHPYLM.Sentence,Int64,Int64,NHPYLM.IntContainer,NHPYLM.IntContainer}'>#</a>
**`NHPYLM.viterbi_argmax_backward_sample_k_and_j_to_eos`** &mdash; *Method*.



This method is called when we know the third gram is EOS, so we're only sampling the first gram and second gram.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L508' class='documenter-source'>source</a><br>

<a id='NHPYLM.viterbi_decode-Tuple{NHPYLM.Sampler,NHPYLM.Sentence}' href='#NHPYLM.viterbi_decode-Tuple{NHPYLM.Sampler,NHPYLM.Sentence}'>#</a>
**`NHPYLM.viterbi_decode`** &mdash; *Method*.



This function uses viterbi algorithm to sample the segmentation of a sentence, instead of the approach in the `blocked_gibbs_segment` function above. They should both be valid approaches.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/e6f8ac881f9871e5487f6b618903ccfdd2c249bd/julia-nhpylm/src/Sampler.jl#L622' class='documenter-source'>source</a><br>

