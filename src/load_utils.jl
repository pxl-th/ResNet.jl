function rebuild_conv!(dst, src)
    shape = dst |> size
    filter_x, filter_y = shape[1:2] .+ 1
    for (i, j, k, m) in Iterators.product([1:s for s in shape]...)
        dst[filter_x - i, filter_y - j, k, m] = src[m, k, j, i]
    end
end

function _load_entry!(entry::Chain, params)
    rebuild_conv!(entry[1].weight, params["conv1.weight"])
    copyto!(entry[2].γ, params["bn1.weight"])
    copyto!(entry[2].β, params["bn1.bias"])
    copyto!(entry[2].μ, params["bn1.running_mean"])
    copyto!(entry[2].σ², params["bn1.running_var"])
end

function _load_basic_layer!(basic::Chain, params, layer_id::Int)
    for repeat in 1:length(basic)
        skipcon = basic[repeat][1]
        layer = skipcon.layers
        lk = "layer$layer_id.$(repeat - 1)."

        rebuild_conv!(layer[1].weight, params[lk * "conv1.weight"])
        copyto!(layer[2].γ, params[lk * "bn1.weight"])
        copyto!(layer[2].β, params[lk * "bn1.bias"])
        copyto!(layer[2].μ, params[lk * "bn1.running_mean"])
        copyto!(layer[2].σ², params[lk * "bn1.running_var"])

        rebuild_conv!(layer[3].weight, params[lk * "conv2.weight"])
        copyto!(layer[4].γ, params[lk * "bn2.weight"])
        copyto!(layer[4].β, params[lk * "bn2.bias"])
        copyto!(layer[4].μ, params[lk * "bn2.running_mean"])
        copyto!(layer[4].σ², params[lk * "bn2.running_var"])

        if skipcon.connection isa Shortcut
            connection = skipcon.connection.s
            lk = "layer$layer_id.0.downsample."
            rebuild_conv!(connection[1].weight, params[lk * "0.weight"])
            copyto!(connection[2].γ, params[lk * "1.weight"])
            copyto!(connection[2].β, params[lk * "1.bias"])
            copyto!(connection[2].μ, params[lk * "1.running_mean"])
            copyto!(connection[2].σ², params[lk * "1.running_var"])
        end
    end
end

function _load_layers!(layers::Chain, params, model_size::Int)
    for i in 1:length(layers)
        if model_size < 50
            _load_basic_layer!(layers[i], params, i)
        else
            throw("Not implemented")
        end
    end
end

function from_pretrained(;
    model_size::Int, cache_dir::Union{String, Nothing} = nothing,
    kwargs...,
)
    # url_base = "https://download.pytorch.org/models/"
    url_base = "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/"
    url_map = Dict(
        18 => "semi_supervised_resnet18-d92f0530.pth",
        50 => "semi_supervised_resnet50-08389792.pth",
        # 18 => "resnet18-5c106cde.pth",
        # 34 => "resnet34-333f7ec4.pth",
        # 101 => "resnet101-5d3b4d8f.pth",
        # 152 => "resnet152-b121ed2d.pth",
    )

    if !(model_size in keys(url_map))
        error(
            "Invalid model model_size: $model_size. " *
            "Supported pretrained models: $(keys(url_map)) "
        )
    end

    if cache_dir ≡ nothing
        cache_dir = joinpath(homedir(), ".cache", "ResNet.jl")
        !isdir(cache_dir) && mkdir(cache_dir)
        @info "Using default cache dir $cache_dir"
    end

    params_file = url_map[model_size]
    params_path = joinpath(cache_dir, params_file)
    if !isfile(params_path)
        download_url = url_base * params_file
        @info(
            "Downloading ResNet$model_size params:\n" *
            "\t- from URL: $download_url \n" *
            "\t- to directory: $params_path"
        )
        download(download_url, params_path)
        @info "Finished downloading params."
    end

    params = Pickle.Torch.THload(params_path)
    model = ResNetModel(;size=model_size, kwargs...)

    _load_entry!(model.entry, params)
    _load_layers!(model.layers, params, model.size)
    if model.head ≢ nothing
        copyto!(model.head[end].weight, params["fc.weight"])
        copyto!(model.head[end].bias, params["fc.bias"])
    end

    model
end
