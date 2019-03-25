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
5. To evaluate a model, run `JuliaNhpylm.evaluate("test-data-path", "previously-saved-model-path")`

For steps 3 to 6, I've alternatively provided two scripts `Train.jl` and `Eval.jl` which can be directly invoked from the command line:

1. First, install the `ArgParse` package with `Pkg.add("ArgParse")` in a Julia session. This is a one-time operation.
2. Afterwards, to train a model, run `julia Train.jl --corpus training-data-path --order order --iter iterations --output model-output-path`, for example:

    `julia Train.jl --corpus ../data/brown/basic-train.txt --order 3 --iter 100 --output testmodel`

3. To evaluate a model, run `julia Eval.jl --corpus test-data-path --model previously-saved-model-path`, for example:

`julia Eval.jl --corpus ../data/brown/basic-test.txt --model testmodel`

Note that unlike when invoking the functions in the REPL, there's no need to enclose paths with quotes.

(N.B.: Currently Julia [doesn't yet fully support compilation to binary executables](https://stackoverflow.com/questions/50608970/if-a-julia-script-is-run-from-the-command-line-does-it-need-to-be-re-compiled-e). The support is experimental and I haven't tried it on this project. One can still choose to run the program from the command line, but it will need to be compiled anew every time it's run again, which will take some time. Therefore, the most popular practice is to use one REPL session to load the module first, and perform all the work without leaving the session. Of course, it's also possible to restart the REPL session to continue the work, after the model file has been saved.)
