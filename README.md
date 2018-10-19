# HPYLM Word Segmentation

## Running the program

The program is contained in the folder `julia-hpylm`. It is implemented in the Julia programming language. Please first install Julia following the instructions at https://julialang.org/downloads/

Please note that currently Julia [doesn't yet fully support compilation to binary executables](https://stackoverflow.com/questions/50608970/if-a-julia-script-is-run-from-the-command-line-does-it-need-to-be-re-compiled-e). The support is experimental and I haven't tried it on this project. One can still choose to run the program from the command line, but it will need to be compiled anew every time it's run again, which will take some time. Therefore, the most popular practice is to use one REPL session to load the module first, and perform all the work without leaving the session. Of course, it's also possible to restart the REPL session to continue the work, after the model file has been saved.

To run the program (with Julia version >= 1.0):

1. launch a Julia REPL in the project root folder with the `julia` command.
2. You may need to install some packages first with the following commands:
   ```
   using Pkg
   Pkg.add("SpecialFunctions")
   Pkg.add("Distributions")
   Pkg.add("Pipe")
   Pkg.build("Arpack")
   ```
3. Run `include("HPYLM.jl")`
4. To train a model, run `HPYLM.train("training-data-path", ngram-size, iterations, "model-output-path")`
5. To evaluate a model, run `HPYLM.evaluate("test-data-path", "previously-saved-model-path")`

For steps 3 to 6, I've alternatively provided two scripts `Train.jl` and `Eval.jl` which can be directly invoked from the command line:

1. First, install the `ArgParse` package with `Pkg.add("ArgParse")` in a Julia session. This is a one-time operation.
2. Afterwards, to train a model, run `julia Train.jl --corpus training-data-path --order order --iter iterations --output model-output-path`, for example:

`julia Train.jl --corpus ../data/brown/basic-train.txt --order 3 --iter 100 --output testmodel`

3. To evaluate a model, run `julia Eval.jl --corpus test-data-path --model previously-saved-model-path`, for example:

`julia Eval.jl --corpus ../data/brown/basic-test.txt --model testmodel`

Note that unlike when invoking the functions in the REPL, there's no need to enclose paths with quotes.

## Motivation, method, hypotheses

Teh (2006) proposed a Bayesian language model based on hierarchical Pitman-Yor process, which can be used to construct n-gram models effectively. The Pitman-Yor process is able to produce power-law distributions which are observed in natural languages. The model approximates interpolated Kneser-Ney smoothing for n-gram models.

Mochihashi et al. (2009) applied the model on Chinese/Japanese text segmentation by essentially treating a word as an n-gram of individual characters. They reported significantly better results than previous unsupervised segmentation of Chinese and Japanese. They also reported being able to easily modify the model to incorporate elements of semi-supervised or completely supervised learning, which further improved accuracy.

Granted, the state-of-the-art results on language modeling and word segmentation are still achieved by supervised learning methods, just as is the case with many other tasks. However, unsupervised learning methods could still be interesting, especially on languages for which transcriptions/gold standard data are lacking or inherently harder to obtain.

The minimum expectation would be to successfully build the hierarchical Bayesian model as described by Teh (2006). Testing would be performed at least on the publicly available AP News data, which is originally used by Teh, and the Brown corpus and the State of the Union corpus, which are used by Dr. Dyer and his colleagues in their testing. Additional testing on different languages might also be performed to observe differences in performances.

Currently, the plan is to first understand the implementation by Victor Chahuneau and Dr. Chris Dyer, and then write my own implementation in another language. The implementation is currently being done in Rust.

## Relevant literature

- Teh (2006) [
  A hierarchical Bayesian language model based on Pitman-Yor processes
  ](https://dl.acm.org/citation.cfm?id=1220299)

  This is the original paper which proposed and clearly described the model.

- Mochihashi et al. (2009) [
  Bayesian unsupervised word segmentation with nested Pitman-Yor language modeling
  ](https://dl.acm.org/citation.cfm?id=1687894)

  The second paper reported the actual application of the model on Chinese segmentation. However, its explanation of the hierarchical Pitman-Yor language model itself doesn't seem to be very clear. The original paper by Teh seems to do a better job.

## Available data, tools, resources

- Data

  - [AP News](https://ibm.ent.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4)
  - [Brown & State of the Union corpus](http://demo.clab.cs.cmu.edu/cdyer/dhpyplm-data.tar.gz)
  - [the 4th SIGHAN workshop](http://sighan.cs.uchicago.edu/bakeoff2005/)

- Tools

  There are already several implementations of the Pitman Yor process. However, all of them differ from the ideas from the Mochihashi et al. (2009) paper in some way and cannot be directly applied towards Chinese segmentation.

  - Dr. Chris Dyer from Carnegie Mellon University has [an implementation in C++](https://github.com/redpony/cpyp) of Pitman-Yor process partly based on the Teh (2006) paper, while his student Victor Chahuneau has a similar [implementation in Python](https://github.com/vchahun/vpyp). However, they didn't implement the nested hierarchical Pitman-Yor model as proposed by the Mochihashi et al. (2009) paper. Also, as their focus is on language models such as Latent Pitman-Yor allocation topic model, the projects don't contain code for word segmentation. My Rust implementation will be based on their project structure.
  - The projects [LatticeWordSegmentation](https://github.com/fgnt/LatticeWordSegmentation) and [nhpylm](https://github.com/fgnt/nhpylm) by Dr. Oliver Walter from Universität Paderborn and his colleagues contain an implementation of the nested hierarchical Pitman-Yor model. However, their focus is on speech processing based on phonemes, lattices and finite state transducers, instead of text processing, and the sample code doesn't seem to work on any UTF-8 text input data.

## Project members

- Xiang Ji (x-ji)
