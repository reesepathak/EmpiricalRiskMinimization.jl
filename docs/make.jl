using Documenter, EmpiricalRiskMinimization

makedocs(modules=[EmpiricalRiskMinimization],
         doctest=false, clean=true,
         format=:html,
         sitename="EmpiricalRisk Minimization.jl",
         authors="Reese Pathak, Guillermo Angeris, Sanjay Lall, Stephen Boyd.",
         pages=Any[
             "Home" => "index.md",
             "Basic Usage" => Any["usage/introduction.md", "usage/installation.md"],
             "Examples" => Any["examples/index.md"],
             "Functions" => Any["functions/losses.md", "functions/regularizers.md"]]
         )
deploydocs(repo = "github.com/reesepathak/EmpiricalRiskMinimization.jl.git",
           target = "build", 
	   julia = "0.6", 
	   osname = "linux",
           deps = nothing,
           make = nothing)
