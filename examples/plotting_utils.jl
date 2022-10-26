export plot_image, plot_loss,plot_3d_slices
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


function plot_3d_slices(q_array, fig; cmap=nothing)
    counter_i = 1
    clim = nothing 
    num_cols = 7
    num_slices = num_cols
    slice_range = div(64,num_slices)
    for i in 1:num_slices
        slice_i = (i-1)*slice_range +1
        subplot(2,num_cols,counter_i); title("Lateral slice $(slice_i)/64")
        img = imshow(q_array[:,slice_i,:]', interpolation="none", cmap=cmap, clim=clim)# vmin=vmin, vmax=vmax ) 
        if i == 1
            clim = img.get_clim()
            ylabel("Depth z [grid points]")
        else
            yticks([])
        end
        xticks([])
        counter_i += 1
    end

    for i in 1:num_slices
        slice_i = (i-1)*slice_range +1
        subplot(2,num_cols,counter_i); title("Top-down slice $(slice_i)/64")
        img = imshow(q_array[:,:,slice_i]', interpolation="none", cmap=cmap, clim=clim)# vmin=vmin, vmax=vmax ) 
        if i == 1
            clim = img.get_clim()
            ylabel("Lateral x [grid points]")
        else
            yticks([])
        end
        xlabel("Lateral y [grid points]")
        counter_i += 1
    end

    return fig 
end

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