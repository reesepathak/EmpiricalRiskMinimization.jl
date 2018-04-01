using Documenter, EmpiricalRiskMinimization

makedocs(modules=[EmpiricalRiskMinimization], doctest=true)

deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"), 
	   repo = "github.com/reesepathak/EmpiricalRiskMinimization.jl", 
	   julia = "0.6.2", 
	   osname = "linux")
