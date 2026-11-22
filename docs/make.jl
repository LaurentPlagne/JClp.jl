push!(LOAD_PATH,joinpath(@__DIR__, ".."))
using Documenter, JClp

makedocs(
    modules = [JClp],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Laurent Plagne",
    sitename = "JClp.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com//JClp.jl.git",
)
