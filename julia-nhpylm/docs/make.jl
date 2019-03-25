push!(LOAD_PATH, "../src/")
# Don't know why but apparently the `push!` line wasn't working, but the `include` line is.
# include("../src/Model.jl")
using Documenter, JuliaNhpylm
using DocumenterLaTeX
using DocumenterMarkdown
# using Documenter

makedocs(
    # format = LaTeX(platform = "docker"),
    # format = LaTeX(),
    format = Markdown(),
    sitename = "Nested Pitman-Yor Language Model"
    )