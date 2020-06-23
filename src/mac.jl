macro propagate(x, fun)
    fs = esc(fun)
    xs = esc(x)
    @show fs
    @show xs
    quote
        r = $fs($xs)
        @show r $fs $xs
        $fs(r)
    end
end
