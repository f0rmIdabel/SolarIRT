"""
File from Tiago,

changed from (x,y,z) to (z,x,y) to fit with convention used in rest of project
"""

using Unitful
using Images: warp, indices_spatial
using CoordinateTransformations, Interpolations



function rotate_data!(data::Array, ϕ::Float64)

    if ϕ ≈ 0
    elseif ϕ ≈ π/2
        data = permutedims(data, [1,3,2])
        data = reverse(data, dims = 2)
    elseif ϕ ≈ π
        data = reverse(data, dims=2)
        data = reverse(data, dims=3)
    elseif ϕ ≈ 3π/2
        data = permutedims(data, [1,3,2])
        data = reverse(data, dims = 3)
    else
        println("ϕ = ", ϕ, " is not a valid angle.")
    end

    return data
end

"""
    shift_image(image::Array, shift_x::Real, shift_y::Real)

Shift a 2D array by an amount of `shift_x` and `shift_y` pixels,
in the first and seecond dimensions. Assumes the image is horizontally
periodic, so the returned array has the same dimensions as `image`.
Tiago
"""
function shift_image(image::Array, shift_x::Real, shift_y::Real)
    transl = Translation(shift_x, shift_y)

    u = unit(image[1,1])
    image = ustrip(image)

    warp(image, transl, indices_spatial(image), Linear(), Periodic())
    image *= u
end


"""
    shift_variable!(var::Array, height::Array{<:Unitful.Length, 1},
                         pixel_size::Unitful.Length, μ::Real)

Shift (or translate) a 3D array that is horizontally periodic in the first two dimensions
according to a polar angle θ given by μ = cos(θ).

This function is easier to understand, but about 5x slower than `translate!`.
Tiago
"""
function shift_variable!(var::Array, height::Array{<:Unitful.Length, 1},
                         pixel_size::Unitful.Length, μ::Real)
    θ = acos(μ)
    shift_pix = uconvert.(Unitful.NoUnits, height .* tan(θ) ./ pixel_size)
    for i=1:length(height)
        var[i,:, :] .= shift_image(var[i, :, :], shift_pix[i], 0.)
    end
end
