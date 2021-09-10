using PyPlot

rc("font", family="serif", size=12)
font_prop = matplotlib.font_manager.FontProperties(
    family="serif",
    style="normal",
    size=13
)
sfmt=matplotlib.ticker.ScalarFormatter(useMathText=true)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

function plot_image(image::Array{T,2}; figsize, vmin=nothing, vmax=nothing, cmap=nothing, title, path="./figs/image.png", xlabel="", ylabel="") where T

    (vmin === nothing) && (vmin = min(image...))
    (vmax === nothing) && (vmax = max(image...))
    fig = figure(figsize=figsize)
    imshow(image, vmin=vmin, vmax=vmax, aspect=1, resample=true, interpolation="none", filterrad=1, cmap=cmap)
    PyPlot.title(title)
    # colorbar(fraction=0.047, pad=0.01, format=sfmt)
    colorbar(fraction=0.047, pad=0.01)
    grid(false)
    PyPlot.xlabel(xlabel)
    PyPlot.ylabel(ylabel)
    savefig(path, format="png", bbox_inches="tight", dpi=200, pad_inches=.05)
    close(fig)

end

function plot_loss(n, loss; figsize, color, title, path="./figs/image.png", xlabel="", ylabel="")

    fig = figure(figsize=figsize)
    plot(n, loss, color=color)
    PyPlot.title(title)
    PyPlot.ylabel(ylabel)
    PyPlot.xlabel(xlabel)
    grid(true, which="major")
    grid(true, which="minor")
    savefig(path, format="png", bbox_inches="tight", dpi=200, pad_inches=.05)
    close(fig)

end