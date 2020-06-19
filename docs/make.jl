using ResNet
using Documenter

makedocs(;
    modules=[ResNet],
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/ResNet.jl/blob/{commit}{path}#L{line}",
    sitename="ResNet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pxl-th.github.io/ResNet.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pxl-th/ResNet.jl",
)
