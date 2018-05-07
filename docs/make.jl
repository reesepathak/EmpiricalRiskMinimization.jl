using Documenter, EmpiricalRiskMinimization

push!(LOAD_PATH, "../src/")

makedocs(modules=[EmpiricalRiskMinimization],
         repo="http://github.com/reesepathak/EmpiricalRiskMinimization.jl.git",
         doctest=false, clean=true, debug=true,
         format=:html,
         sitename="EmpiricalRisk Minimization.jl",
         authors="Sanjay Lall, Reese Pathak, Guillermo Angeris, Stephen Boyd.",
         checkdocks=:exported,
         pages=Any[
             "Home" => "index.md",
             "Examples" => Any["examples/walkthrough.md", "examples/additional_examples.md"],
             "Usage" => Any["usage/models.md", "usage/losses.md", "usage/regularizers.md", "usage/validation.md", "usage/prediction.md"],
             "Library Reference" => Any["lib/models.md", "lib/losses.md", "lib/regularizers.md",
                                        "lib/validation.md", "lib/prediction.md", "lib/convenience.md"]])
deploydocs(repo = "github.com/reesepathak/EmpiricalRiskMinimization.jl.git",
           target = "build", 
	   julia = "0.6", 
	   osname = "linux",
           deps = nothing,
           make = nothing)
