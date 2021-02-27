include("shift_tools.jl")
include("radiation.jl")
using FastGaussQuadrature

"""
1.5D feautrier calculation.
"""
function feautrier(S, α, z, nμ::Int64, nφ::Int64, pixel_size)

    # ===================================================================
    # SET UP VARIABLES
    # ===================================================================
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5
    nz, nx, ny = size(S)

    D = Array{Float64,3}(undef,nz-1,nx,ny)
    E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
    p = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"
    P = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"
    J = zeros(nz,nx,ny)u"kW / m^2 / sr / nm"

    # ==================================================================
    # FEAUTRIER
    # ==================================================================

    for m=1:nμ

        # ϕ = 0
        #########################################
        S_ = copy(S)
        α_ = copy(α)
        shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
        shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
        α_ /= μ[m]
        S_ /= μ[m]

        τ = optical_depth(α_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        P += p

        # ϕ = π
        #########################################
        S_ = copy(S)
        α_ = copy(α)

        S_ = reverse(S_, dims = 2)
        S_ = reverse(S_, dims = 3)
        α_ = reverse(α_, dims = 2)
        α_ = reverse(α_, dims = 3)

        shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
        shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
        α_ /= μ[m]
        S_ /= μ[m]

        τ = optical_depth(α_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        p = reverse(p, dims = 2)
        p = reverse(p, dims = 3)
        P += p

        # ϕ = 3π/2
        #########################################
        S_ = copy(S)
        α_ = copy(α)

        S_ = permutedims(S_, [1,3,2])
        S_ = reverse(S_, dims = 3)
        α_ = permutedims(α_, [1,3,2])
        α_ = reverse(α_, dims = 3)

        shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
        shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
        α_ /= μ[m]
        S_ /= μ[m]

        τ = optical_depth(α_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        p = permutedims(p, [1,3,2])
        p = reverse(p, dims = 3)
        P += p

        # ϕ = π/2
        ############################################
        S_ = copy(S)
        α_ = copy(α)

        S_ = permutedims(S_, [1,3,2])
        S_ = reverse(S_, dims = 2)
        α_ = permutedims(α_, [1,3,2])
        α_ = reverse(α_, dims = 2)

        shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
        shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
        α_ /= μ[m]
        S_ /= μ[m]

        τ = optical_depth(α_, z)
        p[end,:,:] = forward(D, E, S_, τ, μ[m])
        backward(p, D, E)
        reverse(p, dims = 2)
        p = permutedims(p, [1,3,2])
        p = reverse(p, dims = 2)
        P += p

        # Shift back μ
        ################################################
        shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
        shift_variable!(P, z[1:end-1], pixel_size, -1.0)

        # Add to J
        J = J .+ w[m]*P/nφ
    end

    return J
end

"""
From Tiago, translated from Python to Julia
Computes S and J from a lambda iteration.
"""
function lambda_iteration(atmosphere::Atmosphere, atom::Atom, radiation::Radiation, nμ=3, nϕ=4, max_iterations=100)

    nμ, nϕ = get_angles()
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5

    # ==================================================================
    # ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    z = atmosphere.z
    temperature = atmosphere.temperature
    pixel_size = abs(x[2] - x[1])

    # ==================================================================
    # RADIATION DATA
    # ==================================================================
    λ = atom.λ
    α = radiation.α
    B = radiation.B
    nλ, nz, nx, ny = size(α)

    # ==================================================================
    # SET UP TIMER
    # ==================================================================
    time = Array{Float64,1}(undef, nλ)

    # ===================================================================
    # CALCULATE BB SOURCE FUNCTION
    # ===================================================================
    J = Array{Float64,4}(undef, nλ, nz, nx, ny)u"kW / m^2 / sr / nm"


    Threads.@threads for l=1:nλ

        D = Array{Float64,3}(undef,nz-1,nx,ny)
        E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
        p = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
        P = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
        Jλ = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"

        S = B[l,:,:,:]
        fill!(p, 0.0u"kW / m^2 / sr / nm")
        fill!(P, 0.0u"kW / m^2 / sr / nm")
        fill!(Jλ, 0.0u"kW / m^2 / sr / nm")

        et = @elapsed for m=1:nμ

            # ϕ = 0
            #########################################
            S_ = copy(S)
            α_ = copy(α)
            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            P += p

            # ϕ = π
            #########################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = reverse(S_, dims = 2)
            S_ = reverse(S_, dims = 3)
            α_ = reverse(α_, dims = 2)
            α_ = reverse(α_, dims = 3)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            p = reverse(p, dims = 2)
            p = reverse(p, dims = 3)
            P += p

            # ϕ = 3π/2
            #########################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = permutedims(S_, [1,3,2])
            S_ = reverse(S_, dims = 3)
            α_ = permutedims(α_, [1,3,2])
            α_ = reverse(α_, dims = 3)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            p = permutedims(p, [1,3,2])
            p = reverse(p, dims = 3)
            P += p

            # ϕ = π/2
            ############################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = permutedims(S_, [1,3,2])
            S_ = reverse(S_, dims = 2)
            α_ = permutedims(α_, [1,3,2])
            α_ = reverse(α_, dims = 2)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            reverse(p, dims = 2)
            p = permutedims(p, [1,3,2])
            p = reverse(p, dims = 2)
            P += p

            # Shift back μ
            ################################################
            shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
            shift_variable!(P, z[1:end-1], pixel_size, -1.0)

            # Add to J
            Jλ = Jλ .+ w[m]*P/nφ
        end

        J[l,:,:,:] = et.value
        time[l] = et.time
    end

    # ==================================================================
    # WRITE TO FILE
    # ==================================================================
    h5open("../out/output.h5", "w") do file
        write(file, "J", ustrip(J))
        write(file, "time", time)
    end
end

function feautrier(atmosphere, atom, radiation)
    # ===================================================================
    # LOAD ANGLES
    # ===================================================================
    nμ, nϕ = get_angles()
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5

    # ==================================================================
    # LOAD ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    z = atmosphere.z
    pixel_size = abs(x[2] - x[1])

    # ==================================================================
    # RADIATION DATA
    # ==================================================================
    λ = atom.λ
    α = radiation.α
    B = radiation.B
    nλ, nz, nx, ny = size(α)

    # ==================================================================
    # SET UP TIMER
    # ==================================================================
    time = Array{Float64,1}(undef, nλ)

    # ==================================================================
    # SET UP ARRAYS
    # ==================================================================
    D = Array{Float64,3}(undef,nz-1,nx,ny)
    E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
    p = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    P = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    J = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"

    # ==================================================================
    # FEAUTRIER
    # ==================================================================

    for l=1:nλ

        fill!(p, 0.0u"kW / m^2 / sr / nm")
        fill!(P, 0.0u"kW / m^2 / sr / nm")
        fill!(J, 0.0u"kW / m^2 / sr / nm")

        S = B[l,:,:,:]
        α = α[l,:,:,:]

        for m=1:nμ

            # ϕ = 0
            #########################################
            S_ = copy(S)
            α_ = copy(α)
            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            P += p

            # ϕ = π
            #########################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = reverse(S_, dims = 2)
            S_ = reverse(S_, dims = 3)
            α_ = reverse(α_, dims = 2)
            α_ = reverse(α_, dims = 3)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            p = reverse(p, dims = 2)
            p = reverse(p, dims = 3)
            P += p

            # ϕ = 3π/2
            #########################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = permutedims(S_, [1,3,2])
            S_ = reverse(S_, dims = 3)
            α_ = permutedims(α_, [1,3,2])
            α_ = reverse(α_, dims = 3)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            p = permutedims(p, [1,3,2])
            p = reverse(p, dims = 3)
            P += p

            # ϕ = π/2
            ############################################
            S_ = copy(S)
            α_ = copy(α)

            S_ = permutedims(S_, [1,3,2])
            S_ = reverse(S_, dims = 2)
            α_ = permutedims(α_, [1,3,2])
            α_ = reverse(α_, dims = 2)

            shift_variable!(S_, z[1:end-1], pixel_size, μ[m])
            shift_variable!(α_, z[1:end-1], pixel_size, μ[m])
            α_ /= μ[m]
            S_ /= μ[m]

            τ = optical_depth(α_, z)
            p[end,:,:] = forward(D, E, S_, τ, μ[m])
            backward(p, D, E)
            reverse(p, dims = 2)
            p = permutedims(p, [1,3,2])
            p = reverse(p, dims = 2)
            P += p

            # Shift back μ
            ################################################
            shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
            shift_variable!(P, z[1:end-1], pixel_size, -1.0)

            # Add to J
            J = J .+ w[m]*P/nφ
        end
    end


    return J
end

"""
Forward-propagation to find the coefficients.
"""
function forward(D::Array{Float64, 3},
                 E::Array{<:Unitful.Quantity,3},
                 S::Array{<:Unitful.Quantity,3},
                 τ::Array{Float64, 3},
                 μ::Float64)

    nz, nx, ny =  size(τ)
    P_end = Array{Float64,2}(undef,nx,ny)u"kW / m^2 / sr / nm"

    for j=1:ny
        for i=1:nx
            Δτ = τ[2:end,i,j] .- τ[1:end-1,i,j]

            # From boundary condition at the top
            E[1,i,j] = 0.0u"kW / m^2 / sr / nm"
            D[1,i,j] = 1.0/(Δτ[1]/μ + 1.0)

            #forward
            for k=2:length(Δτ)
                A = 2μ^2 / (Δτ[k-1]*(Δτ[k-1] + Δτ[k]))
                B = 1.0 + 2μ^2 / (Δτ[k]*Δτ[k-1])        #should use steins trick here
                C = 2μ^2 /(Δτ[k]*(Δτ[k-1] + Δτ[k]))

                D[k,i,j] = C / (B - A*D[k-1,i,j])
                E[k,i,j] = (S[k,i,j] + A*E[k-1,i,j]) / (B - A*D[k-1,i,j])
            end

            # From boundary
            P_end[i,j] = ( E[end,i,j] + Δτ[end]/μ * S[end,i,j] + (S[end,i,j] - S[end-1,i,j]) ) / (1.0 - D[end,i,j] + Δτ[end]/μ)
        end
    end

    return P_end
end

"""
Back-propagation to find the P.
"""
function backward(P::Array{<:Unitful.Quantity,3},
                  D::Array{Float64, 3},
                  E::Array{<:Unitful.Quantity,3})

    nz, nx, ny = size(D)

    for j=1:ny
        for i=1:nx
            for k in range(nz, step=-1,stop=1)
                P[k,i,j] = D[k,i,j]*P[k+1,i,j] + E[k,i,j]
            end
        end
    end
end
