function from_pretrained(
    model_name::String, cache_dir::Union{String, Nothing} = nothing
)
    url_base = "https://download.pytorch.org/models/"
    url_map = Dict(
        "resnet18" => "resnet18-5c106cde.pth",
        "resnet34" => "resnet34-333f7ec4.pth",
        "resnet50" => "resnet50-19c8e357.pth",
        "resnet101" => "resnet101-5d3b4d8f.pth",
        "resnet152" => "resnet152-b121ed2d.pth",
    )
end
