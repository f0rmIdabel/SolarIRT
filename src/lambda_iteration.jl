include("feautrier.jl")
include("radiation.jl")
using LinearAlgebra

"""
Computes S and J from a lambda iteration.
"""
function lambda_iteration(atmosphere::Atmosphere, radiation::Radiation, nμ=3, nϕ=4, max_iterations=100)
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
    λ = radiation.λ
    α = radiation.α
    ε = radiation.ε

    nλ, nz, nx, ny = size(α)

    # ==================================================================
    # SET UP TIMER
    # ==================================================================

    time = Array{Float64,2}(undef, max_iterations, nλ)
    error = Array{Float64,2}(undef, max_iterations, nλ)

    # ===================================================================
    # CALCULATE BB SOURCE FUNCTION
    # ===================================================================

    J = Array{Float64,4}(undef, nλ, nz, nx, ny)u"kW / m^2 / sr / nm"
    B = Array{Float64,4}(undef, nλ, nz, nx, ny)u"kW / m^2 / sr / nm"
    S = Array{Float64,4}(undef, nλ, nz, nx, ny)u"kW / m^2 / sr / nm"

    Threads.@threads for l=1:nλ
        B[l,:,:,:] = blackbody_lambda.(λ[l], temperature)
    end

    S = copy(B)
    println("--Starting λ-iteration.....................\n")
    for n=1:max_iterations
        print("--Iteration ", n, "..............................")

        Threads.@threads for l=1:nλ
            f = @timed feautrier(S[l,:,:,:], α[l,:,:,:], z, nμ, nϕ, pixel_size)
            J[l,:,:,:] = f.value
            time[n,l] = f.time
        end

        S_new = (1.0 .- ε) .* J + ε .* B

        converged = check_converging(S, S_, error, n)

        if converged
            println("--Convergence at iteration n = ", n, ". λ-iteration finished.")
            S = S_new
            time = time[1:n,:]
            error = error[1:n,:]
            break
        else
            S = S_new
        end
    end

    # ==================================================================
    # WRITE TO FILE
    # ==================================================================
    out = h5open("../../out/output_integral.h5", "w")
    write(out, "lambda", ustrip(λ))
    write(out, "J", ustrip(J))
    write(out, "S", ustrip(S))
    write(out, "B", ustrip(B))
    write(out, "time", time)
    write(out, "error", error)
    close(out)
end

"""
Check if the relative difference between two arrays
S1 and S2 is smaller than a given criterion.
"""
function check_converged(S1, S2, error, n, criterion = 1e-4)
    nλ, nz, nx, ny = size(S1)
    err = Array{Float64, 1}(undef, nλ)
    for l=1:nλ
        err[l] = norm( abs.(S1[l,:,:,:] .- S2[l,:,:,:]) ./S1[l,:,:,:] )
    end

    error[n,:] = err
    println(@sprintf("Relative error = %.2e.", maximum(err)))

    converged = false

    if maximum(err) < criterion
        converged = true
    end

    return converged
end





"""
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
