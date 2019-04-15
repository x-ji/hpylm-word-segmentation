## Running the program

You may [install Julia here](https://julialang.org/downloads/).

To run the program (with Julia version >= 1.0):

1. launch a Julia REPL in this folder (`julia-nhpylm`) with the `julia` command.
   (N.B.: If the REPL wasn't launched in that folder, you may execute `cd("julia-nhpylm")` to switch the current working directory.)
2. You may need to install some packages first with the following commands:
    ```julia
    using Pkg
    Pkg.add("LegacyStrings")
    Pkg.add("Compat")
    Pkg.add("StatsBase")
    Pkg.add("SpecialFunctions")
    Pkg.add("Distributions")
    Pkg.add("OffsetArrays")
    ```
3. Run `include("./src/JuliaNhpylm.jl")`
4. To train a model, run `JuliaNhpylm.train([training-data-path], [model-output-path], [split_proportion], [epochs], [max_word_length])`
5. To use a previously saved model to segment (test) sentences, run `JuliaNhpylm.segment("test-data-path", "previously-saved-model-path", "output-file-path")`. The segmented sentences will be saved to the path supplied.

For evaluation (i.e. obtaining accuracy, precision, F1 scores etc.), please use scripts such as the [one from the SIGHAN Bakeoff challenge](icwb2-data/scripts/score.pl), or [the one](https://homepages.inf.ed.ac.uk/sgwater/software/dpseg-1.2.1.tar.gz) provided by Prof. Goldwater.

(N.B.: Currently Julia [doesn't yet fully support compilation to binary executables](https://stackoverflow.com/questions/50608970/if-a-julia-script-is-run-from-the-command-line-does-it-need-to-be-re-compiled-e). The support is experimental and I haven't tried it on this project. One can still choose to run the program from the command line, but it will need to be compiled anew every time it's run again, which will take some time. Therefore, the most popular practice is to use one REPL session to load the module first, and perform all the work without leaving the session. Of course, it's also possible to restart the REPL session to continue the work, after the model file has been saved.)
