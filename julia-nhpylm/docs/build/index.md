
<a id='Julia-NHPYLM-Documentation-1'></a>

# Julia-NHPYLM Documentation




<a id='Corpus.jl-1'></a>

## Corpus.jl

<a id='JuliaNhpylm.Corpus' href='#JuliaNhpylm.Corpus'>#</a>
**`JuliaNhpylm.Corpus`** &mdash; *Type*.



This struct keeps track of all sentences from the corpus files, and optionally the "true" segmentations, if any.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L27-L29' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.Dataset' href='#JuliaNhpylm.Dataset'>#</a>
**`JuliaNhpylm.Dataset`** &mdash; *Type*.



This struct holds all the structs related to a session/task, including the vocabulary, the corpus and the sentences produced from the corpus.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L65-L67' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.Vocabulary' href='#JuliaNhpylm.Vocabulary'>#</a>
**`JuliaNhpylm.Vocabulary`** &mdash; *Type*.



This struct keeps track of all the characters in the target corpus.

This is necessary because in the character CHPYLM, the G_0 needs to be calculated via a uniform distribution over all possible characters of the target language.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L5-L9' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_sentence-Tuple{JuliaNhpylm.Corpus,LegacyStrings.UTF32String}' href='#JuliaNhpylm.add_sentence-Tuple{JuliaNhpylm.Corpus,LegacyStrings.UTF32String}'>#</a>
**`JuliaNhpylm.add_sentence`** &mdash; *Method*.



Add an individual sentence to the corpus


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L39-L41' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_sentence-Tuple{JuliaNhpylm.Dataset,LegacyStrings.UTF32String,Array{JuliaNhpylm.Sentence,1}}' href='#JuliaNhpylm.add_sentence-Tuple{JuliaNhpylm.Dataset,LegacyStrings.UTF32String,Array{JuliaNhpylm.Sentence,1}}'>#</a>
**`JuliaNhpylm.add_sentence`** &mdash; *Method*.



Add a sentence to the train or dev sentence vector of the dataset


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L155' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.read_corpus-Tuple{JuliaNhpylm.Corpus,IOStream}' href='#JuliaNhpylm.read_corpus-Tuple{JuliaNhpylm.Corpus,IOStream}'>#</a>
**`JuliaNhpylm.read_corpus`** &mdash; *Method*.



Read the corpus from an input stream


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Corpus.jl#L46-L48' class='documenter-source'>source</a><br>


<a id='CType.jl-1'></a>

## CType.jl

<a id='JuliaNhpylm.detect_ctype-Tuple{Char}' href='#JuliaNhpylm.detect_ctype-Tuple{Char}'>#</a>
**`JuliaNhpylm.detect_ctype`** &mdash; *Method*.



Detect the type of a given character


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CType.jl#L283-L285' class='documenter-source'>source</a><br>


<a id='WType.jl-1'></a>

## WType.jl

<a id='JuliaNhpylm.detect_word_type-Tuple{LegacyStrings.UTF32String}' href='#JuliaNhpylm.detect_word_type-Tuple{LegacyStrings.UTF32String}'>#</a>
**`JuliaNhpylm.detect_word_type`** &mdash; *Method*.



Detect the type of a word. Useful when we need to estimate different lambda values for different word types.

Basically it uses a heuristics to see the proportion of character types in this word to assign a word type eventually.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/WType.jl#L138-L142' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.is_dash-Tuple{Char}' href='#JuliaNhpylm.is_dash-Tuple{Char}'>#</a>
**`JuliaNhpylm.is_dash`** &mdash; *Method*.



This refers to the full-width dash used to mark long vowels in Hiragana/Katakana


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/WType.jl#L19' class='documenter-source'>source</a><br>


<a id='Sentence.jl-1'></a>

## Sentence.jl

<a id='JuliaNhpylm.Sentence' href='#JuliaNhpylm.Sentence'>#</a>
**`JuliaNhpylm.Sentence`** &mdash; *Type*.



This struct holds everything that represents a sentence, including the raw string that constitute the sentence, and the potential segmentation of the sentence, if it is segmented, either via a preexisting segmentation or via running the forward-filtering-backward-sampling segmentation algorithm.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Sentence.jl#L5-L7' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_substr_word_id-Tuple{JuliaNhpylm.Sentence,Int64,Int64}' href='#JuliaNhpylm.get_substr_word_id-Tuple{JuliaNhpylm.Sentence,Int64,Int64}'>#</a>
**`JuliaNhpylm.get_substr_word_id`** &mdash; *Method*.



Get the word id of the substring with start*index and end*index. Note that in Julia the end_index is inclusive.

Note that the `hash` method returns UInt! This makes sense because a 2-fold increase in potential hash values can actually help a lot.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/Sentence.jl#L105-L109' class='documenter-source'>source</a><br>


<a id='PYP.jl-1'></a>

## PYP.jl

<a id='JuliaNhpylm.PYP' href='#JuliaNhpylm.PYP'>#</a>
**`JuliaNhpylm.PYP`** &mdash; *Type*.



Each node is essentially a Pitman-Yor process in the hierarchical Pitman-Yor language model

We use a type parameter because it can be either for characters (Char) or for words (UTF32String/UInt)

The root PYP (depth 0) contains zero context. The deeper the depth, the longer the context.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L19-L25' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_customer-Union{Tuple{T}, Tuple{PYP{T},T,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Bool,IntContainer}} where T' href='#JuliaNhpylm.add_customer-Union{Tuple{T}, Tuple{PYP{T},T,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Bool,IntContainer}} where T'>#</a>
**`JuliaNhpylm.add_customer`** &mdash; *Method*.



Adds a customer eating a certain dish to a node.

Note that this method is applicable to both the WHPYLM and the CHPYLM, thus the type parameter.

d*array and θ*array contain the d values and θ values for each depth of the relevant HPYLM (Recall that those values are the same for one single depth.)


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L243-L249' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_customer_to_table-Union{Tuple{T}, Tuple{PYP{T},T,Int64,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,IntContainer}} where T' href='#JuliaNhpylm.add_customer_to_table-Union{Tuple{T}, Tuple{PYP{T},T,Int64,Union{Float64, OffsetArray{Float64,1,AA} where AA<:AbstractArray},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,IntContainer}} where T'>#</a>
**`JuliaNhpylm.add_customer_to_table`** &mdash; *Method*.



The second item returned in the tuple is the index of the table to which the customer is added.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L167' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.compute_p_w_with_parent_p_w-Union{Tuple{T}, Tuple{PYP{T},T,Float64,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray}} where T' href='#JuliaNhpylm.compute_p_w_with_parent_p_w-Union{Tuple{T}, Tuple{PYP{T},T,Float64,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray}} where T'>#</a>
**`JuliaNhpylm.compute_p_w_with_parent_p_w`** &mdash; *Method*.



Compute the possibility of the word/char `dish` being generated from this pyp (i.e. having this pyp as its context). The equation is the one recorded in the original Teh 2006 paper.

When is*parent*p*w == True, the third argument is the parent*p*w. Otherwise it's simply the G*0.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L415-L419' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.find_child_pyp-Union{Tuple{T}, Tuple{PYP{T},T}, Tuple{PYP{T},T,Bool}} where T' href='#JuliaNhpylm.find_child_pyp-Union{Tuple{T}, Tuple{PYP{T},T}, Tuple{PYP{T},T,Bool}} where T'>#</a>
**`JuliaNhpylm.find_child_pyp`** &mdash; *Method*.



Find the child PYP whose context is the given dish


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L149-L151' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_all_pyps_at_depth-Union{Tuple{T}, Tuple{PYP{T},Int64,OffsetArray{PYP{T},1,AA} where AA<:AbstractArray}} where T' href='#JuliaNhpylm.get_all_pyps_at_depth-Union{Tuple{T}, Tuple{PYP{T},Int64,OffsetArray{PYP{T},1,AA} where AA<:AbstractArray}} where T'>#</a>
**`JuliaNhpylm.get_all_pyps_at_depth`** &mdash; *Method*.



If run successfully, this function should put all pyps at the specified depth into the accumulator vector.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L586' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_max_depth-Union{Tuple{T}, Tuple{PYP{T},Int64}} where T' href='#JuliaNhpylm.get_max_depth-Union{Tuple{T}, Tuple{PYP{T},Int64}} where T'>#</a>
**`JuliaNhpylm.get_max_depth`** &mdash; *Method*.



Basically a DFS to get the maximum depth of the tree with this `pyp` as its root


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L522' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_num_nodes-Union{Tuple{PYP{T}}, Tuple{T}} where T' href='#JuliaNhpylm.get_num_nodes-Union{Tuple{PYP{T}}, Tuple{T}} where T'>#</a>
**`JuliaNhpylm.get_num_nodes`** &mdash; *Method*.



A DFS to get the total number of nodes with this `pyp` as the root


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L534' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_num_tables_serving_dish-Union{Tuple{T}, Tuple{PYP{T},T}} where T' href='#JuliaNhpylm.get_num_tables_serving_dish-Union{Tuple{T}, Tuple{PYP{T},T}} where T'>#</a>
**`JuliaNhpylm.get_num_tables_serving_dish`** &mdash; *Method*.



This function explicitly returns the number of **tables** (i.e. not customers) serving a dish!


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L132-L134' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.sample_log_x_u-Union{Tuple{T}, Tuple{PYP{T},Float64}} where T' href='#JuliaNhpylm.sample_log_x_u-Union{Tuple{T}, Tuple{PYP{T},Float64}} where T'>#</a>
**`JuliaNhpylm.sample_log_x_u`** &mdash; *Method*.



Note that only the log of x_u is used in the final sampling, expression (41) of the Teh technical report. Therefore our function also only ever calculates the log. Should be easily refactorable though.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L607-L609' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.sample_summed_y_ui-Union{Tuple{T}, Tuple{PYP{T},Float64,Float64}, Tuple{PYP{T},Float64,Float64,Bool}} where T' href='#JuliaNhpylm.sample_summed_y_ui-Union{Tuple{T}, Tuple{PYP{T},Float64,Float64}, Tuple{PYP{T},Float64,Float64,Bool}} where T'>#</a>
**`JuliaNhpylm.sample_summed_y_ui`** &mdash; *Method*.



Note that in expressions (40) and (41) of the technical report, the yui values are only used when they're summed. So we do the same here.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/PYP.jl#L620-L622' class='documenter-source'>source</a><br>


<a id='HPYLM.jl-1'></a>

## HPYLM.jl

<a id='JuliaNhpylm.CHPYLM' href='#JuliaNhpylm.CHPYLM'>#</a>
**`JuliaNhpylm.CHPYLM`** &mdash; *Type*.



Character Hierarchical Pitman-Yor Language Model 

In this case the HPYLM for characters is an infinite Markov model, different from that used for the words.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L5-L9' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.WHPYLM' href='#JuliaNhpylm.WHPYLM'>#</a>
**`JuliaNhpylm.WHPYLM`** &mdash; *Type*.



Hierarchical Pitman-Yor process for words


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/WHPYLM.jl#L4-L6' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_customer_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}' href='#JuliaNhpylm.add_customer_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}'>#</a>
**`JuliaNhpylm.add_customer_at_index_n`** &mdash; *Method*.



This function adds the customer. See documentation above.

This is a version to be called from the NPYLM.

If the parent*p*w*cache is already set, then update the path*nodes as well.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L107-L113' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.add_customer_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64}' href='#JuliaNhpylm.add_customer_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,Int64}'>#</a>
**`JuliaNhpylm.add_customer_at_index_n`** &mdash; *Method*.



The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you

  * first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass*count or stop*count along the path of the tree
  * sample a new order (depth) according to the conditional probability
  * add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass*count or stop*count along the (new) path

This function adds the customer


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L92-L99' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.compute_log_p_w-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}' href='#JuliaNhpylm.compute_log_p_w-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}'>#</a>
**`JuliaNhpylm.compute_log_p_w`** &mdash; *Method*.



Compute the *log* probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L279-L281' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.compute_p_w-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}' href='#JuliaNhpylm.compute_p_w-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray}'>#</a>
**`JuliaNhpylm.compute_p_w`** &mdash; *Method*.



Compute the probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L271-L273' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.get_num_nodes-Union{Tuple{HPYLM}, Tuple{T}} where T' href='#JuliaNhpylm.get_num_nodes-Union{Tuple{HPYLM}, Tuple{T}} where T'>#</a>
**`JuliaNhpylm.get_num_nodes`** &mdash; *Method*.



Get the total number of nodes in this HPYLM.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/HPYLM.jl#L18' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.init_hyperparameters_at_depth_if_needed-Tuple{JuliaNhpylm.HPYLM,Int64}' href='#JuliaNhpylm.init_hyperparameters_at_depth_if_needed-Tuple{JuliaNhpylm.HPYLM,Int64}'>#</a>
**`JuliaNhpylm.init_hyperparameters_at_depth_if_needed`** &mdash; *Method*.



Sometimes the hyperparameter array can be shorter than the actual depth of the HPYLM (especially for CHPYLM whose depth is dynamic. In this case initialize the hyperparameters at the deeper depth.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/HPYLM.jl#L41' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.sample_depth_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}' href='#JuliaNhpylm.sample_depth_at_index_n-Tuple{JuliaNhpylm.CHPYLM,OffsetArrays.OffsetArray{Char,1,AA} where AA<:AbstractArray,Int64,OffsetArrays.OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArrays.OffsetArray{Union{Nothing, PYP{Char}},1,AA} where AA<:AbstractArray}'>#</a>
**`JuliaNhpylm.sample_depth_at_index_n`** &mdash; *Method*.



Sample the depth of the character at index n of the given characters (word).


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/CHPYLM.jl#L351-L353' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.sample_hyperparameters-Tuple{JuliaNhpylm.HPYLM}' href='#JuliaNhpylm.sample_hyperparameters-Tuple{JuliaNhpylm.HPYLM}'>#</a>
**`JuliaNhpylm.sample_hyperparameters`** &mdash; *Method*.



Sample all hyperparameters of this HPYLM


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/HPYLM.jl#L102' class='documenter-source'>source</a><br>

<a id='JuliaNhpylm.sum_auxiliary_variables_recursively-Union{Tuple{T}, Tuple{HPYLM,PYP{T},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Int64}} where T' href='#JuliaNhpylm.sum_auxiliary_variables_recursively-Union{Tuple{T}, Tuple{HPYLM,PYP{T},OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,OffsetArray{Float64,1,AA} where AA<:AbstractArray,Int64}} where T'>#</a>
**`JuliaNhpylm.sum_auxiliary_variables_recursively`** &mdash; *Method*.



Sum up all values of a auxiliary variable on the same depth into one variable.


<a target='_blank' href='https://github.com/x-ji/hpylm-word-segmentation/blob/00de46b92c706fc34d73339a38621e24364708e9/julia-nhpylm/src/HPYLM.jl#L78' class='documenter-source'>source</a><br>


<a id='Sampler.jl-1'></a>

## Sampler.jl


`$@autodocs Modules = [JuliaNhpylm] Order = [:constant, :type, :function] Pages = ["Sampler.jl"]$

