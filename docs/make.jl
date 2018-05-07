using Documenter, EmpiricalRiskMinimization

makedocs(modules=[EmpiricalRiskMinimization],
         doctest=false, clean=true,
         format=:html,
         sitename="EmpiricalRisk Minimization.jl",
         authors="Reese Pathak, Guillermo Angeris, Sanjay Lall, Stephen Boyd.",
         pages=Any[
             "Home" => "index.md",
             "Examples" => Any["examples/walkthrough.md", "examples/additional_examples.md"],
             "Usage" => Any["usage/models.md", "usage/losses.md", "usage/regularizers.md", "usage/validation.md", "usage/prediction.md"],
             "Library Reference" => Any["lib/models.md", "lib/losses.md", "lib/regularizers.md", "lib/validation.md", "lib/prediction.md"]
         ])
deploydocs(repo = "github.com/reesepathak/EmpiricalRiskMinimization.jl.git",
           target = "build", 
	   julia = "0.6", 
	   osname = "linux",
           deps = nothing,
           make = nothing)
