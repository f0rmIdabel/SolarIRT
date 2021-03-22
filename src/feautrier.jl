include("shift_tools.jl")
include("radiation.jl")


function feautrier(atmosphere::Atmosphere,
                   radiation::RadiationBackground,
                   λ::Unitful.Length,
                   nμ::Int64,
                   nϕ::Int64,
                   output_path::String)

    # ===================================================================
    # ANGLES
    # ===================================================================
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5
    ϕ = collect(0:(nϕ-1)) .* 2π/nϕ

    # ==================================================================
    # ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    z = atmosphere.z
    pixel_size = abs(x[2] - x[1])

    # ===================================================================
    # RADIATION DATA
    # ===================================================================
    α = radiation.α_continuum
    S = radiation.B

    nλ, nz, nx, ny = size(α)

    # ==================================================================
    # SET UP ARRAYS
    # ==================================================================
    D = Array{Float64,3}(undef,nz-1,nx,ny)
    E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
    p = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    P = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    Jμ = Array{Float64,4}(undef,nμ, nz,nx,ny)u"kW / m^2 / sr / nm"

    # ==================================================================
    # FEAUTRIER ROUTINE
    # ==================================================================
    println(@sprintf("--Starting calculation, using %d thread(s)..",
            Threads.nthreads()))

    for λi=1:nλ

        println("\n--[",λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

        #fill!(p, 0.0u"kW / m^2 / sr / nm")

        αλ = α[λi,:,:,:]
        Sλ = S[λi,:,:,:]

        # Create ProgressMeter working with threads
        prc = Progress(nμ*nϕ)
        update!(prc,0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()

        et = @elapsed Threads.@threads for m=1:nμ

            fill!(P, 0.0u"kW / m^2 / sr / nm")

            Sμ = copy(Sλ)
            αμ = copy(αλ)

            shift_variable!(Sμ, z[1:end-1], pixel_size, μ[m])
            shift_variable!(αμ, z[1:end-1], pixel_size, μ[m])

            for i=1:nϕ

                # Advance ProgressMeter
                Threads.atomic_add!(jj, 1)
                Threads.lock(l)
                update!(prc, jj[])
                Threads.unlock(l)

                Sμϕ = copy(Sμ)
                αμϕ = copy(αμ)

                rotate_data!(Sμϕ, ϕ[i])
                rotate_data!(αμϕ, ϕ[i])

                τ = optical_depth(αμϕ, z, μ[m])
                p[end,:,:] = forward(D, E, Sμϕ, τ, 1.0)
                backward(p, D, E)

                rotate_data!(p, ϕ[i])
                P += p
            end

            # Shift back μ
            shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
            shift_variable!(P, z[1:end-1], pixel_size, -1.0)

            # Add to J
            Jμ[m,:,:,:] = w[m]*P/nϕ
        end

        Jλ = sum(Jμ, dims=1)[1,:,:,:]

        h5open(output_path, "r+") do file
            file["J"][λi,:,:,:] = ustrip.(Jλ)
            file["time"][λi] = et
        end
    end
end



function feautrier(atmosphere::Atmosphere,
                   radiation::Radiation,
                   atom::Atom,
                   nμ::Int64,
                   nϕ::Int64,
                   iteration::Int64,
                   output_path::String)

    # ===================================================================
    # ANGLES
    # ===================================================================
    μ, w = gausslegendre(nμ)
    μ = μ ./2.0 .+ 0.5
    ϕ = collect(0:(nϕ-1)) .* 2π/nϕ

    # ==================================================================
    # ATMOSPHERE DATA
    # ==================================================================
    x = atmosphere.x
    z = atmosphere.z
    v = atmosphere.velocity
    pixel_size = abs(x[2] - x[1])

    # ===================================================================
    # RADIATION DATA
    # ===================================================================
    α_continuum = radiation.α_continuum
    ε_continuum = radiation.ε_continuum
    α_line_constant = radiation.α_line_constant
    ε_line = radiation.ε_line
    B = radiation.B

    if iteration == 1
        J = copy(B)
    else
        J = get_Jλ(output_path, iteration-1)
    end

    nλ, nz, nx, ny = size(B)

    # ===================================================================
    # ATOM DATA
    # ===================================================================
    λ = atom.λ
    nλ_bf = atom.nλ_bf
    nλ_bb = atom.nλ_bb
    line = atom.line
    dc = atom.damping_constant
    ΔλD = atom.doppler_width
    λ0 = line.λ0

    # ==================================================================
    # SET UP ARRAYS
    # ==================================================================
    D = Array{Float64,3}(undef,nz-1,nx,ny)
    E = Array{Float64,3}(undef,nz-1,nx,ny)u"kW / m^2 / sr / nm"
    p = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    P = Array{Float64,3}(undef,nz,nx,ny)u"kW / m^2 / sr / nm"
    Jμ = Array{Float64,4}(undef,nμ, nz,nx,ny)u"kW / m^2 / sr / nm"

    # ==================================================================
    # FEAUTRIER ROUTINE
    # ==================================================================
    println(@sprintf("--Starting calculation, using %d thread(s)..",
            Threads.nthreads()))

    for λi=1:2nλ_bf
        println("\n--[",λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

        αλ = α_continuum[λi,:,:,:]
        ελ = ε_continuum[λi,:,:,:]
        Sλ = B[λi,:,:,:] .* ελ .+ J[λi,:,:,:] .* (1.0 .- ελ)

        # Create ProgressMeter working with threads
        prc = Progress(nμ*nϕ)
        update!(prc,0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()

        et = @elapsed Threads.@threads for m=1:nμ

            fill!(P, 0.0u"kW / m^2 / sr / nm")

            αμ = copy(αλ)
            Sμ = copy(Sλ)

            shift_variable!(Sμ, z[1:end-1], pixel_size, μ[m])
            shift_variable!(αμ, z[1:end-1], pixel_size, μ[m])

            for i=1:nϕ

                # Advance ProgressMeter
                Threads.atomic_add!(jj, 1)
                Threads.lock(l)
                update!(prc, jj[])
                Threads.unlock(l)

                Sμϕ = copy(Sμ)
                αμϕ = copy(αμ)

                rotate_data!(Sμϕ, ϕ[i])
                rotate_data!(αμϕ, ϕ[i])

                τ = optical_depth(αμϕ, z, μ[m])
                p[end,:,:] = forward(D, E, Sμϕ, τ, 1.0)
                backward(p, D, E)

                rotate_data!(p, ϕ[i])
                P += p
            end

            # Shift back μ
            shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
            shift_variable!(P, z[1:end-1], pixel_size, -1.0)

            # Add to J
            Jμ[m,:,:,:] = w[m]*P/nϕ
        end

        Jλ = sum(Jμ, dims=1)[1,:,:,:]

        h5open(output_path, "r+") do file
            file["J"][iteration, λi,:,:,:] = ustrip.(Jλ)
            file["time"][iteration, λi] = et
        end
    end

    for λi=(2nλ_bf+1):nλ
        println("\n--[",λi,"/",nλ, "]        ", @sprintf("λ = %.3f nm", ustrip(λ[λi])))

        αλ_continuum = α_continuum[λi,:,:,:]
        ελ_continuum = ε_continuum[λi,:,:,:]
        Bλ = B[λi,:,:,:]
        Jλ = J[λi,:,:,:]

        # Create ProgressMeter working with threads
        prc = Progress(nμ*nϕ)
        update!(prc,0)
        jj = Threads.Atomic{Int}(0)
        l = Threads.SpinLock()

        et = @elapsed Threads.@threads for m=1:nμ

            fill!(P, 0.0u"kW / m^2 / sr / nm")

            for i=1:nϕ

                # Advance ProgressMeter
                Threads.atomic_add!(jj, 1)
                Threads.lock(l)
                update!(prc, jj[])
                Threads.unlock(l)

                v_los = velocity_los(v, μ[m], ϕ[i])
                αλ_line = line_extinction.(λ[λi], λ0, ΔλD, dc, α_line_constant, v_los)
                αλ = αλ_continuum .+ αλ_line
                ελ = (ελ_continuum .* αλ_continuum .+ ε_line .* αλ_line ) ./ αλ

                Sλ = Bλ .* ελ .+ Jλ .* (1.0 .- ελ)


                shift_variable!(Sλ, z[1:end-1], pixel_size, μ[m])
                shift_variable!(αλ, z[1:end-1], pixel_size, μ[m])

                rotate_data!(Sλ, ϕ[i])
                rotate_data!(αλ, ϕ[i])

                τ = optical_depth(αλ, z, μ[m])
                p[end,:,:] = forward(D, E, Sλ, τ, 1.0)
                backward(p, D, E)

                rotate_data!(p, ϕ[i])
                P += p

                ######################## TRICK #######################################
                v_los = velocity_los(v, -μ[m], ϕ[i])
                αλ_line = line_extinction.(λ[λi], λ0, ΔλD, dc, α_line_constant, v_los)
                αλ = αλ_continuum .+ αλ_line
                ελ = (ελ_continuum .* αλ_continuum .+ ε_line .* αλ_line ) ./ αλ
                Sλ = Bλ .* ελ .+ Jλ .* (1.0 .- ελ)

                shift_variable!(Sλ, z[1:end-1], pixel_size, μ[m])
                shift_variable!(αλ, z[1:end-1], pixel_size, μ[m])

                rotate_data!(Sλ, ϕ[i])
                rotate_data!(αλ, ϕ[i])

                τ = optical_depth(αλ, z, μ[m])
                p[end,:,:] = forward(D, E, Sλ, τ, 1.0)
                backward(p, D, E)

                rotate_data!(p, ϕ[i])
                P += p
            end

            # Shift back μ
            shift_variable!(P, z[1:end-1], pixel_size, -μ[m])
            shift_variable!(P, z[1:end-1], pixel_size, -1.0)

            # Add to J
            Jμ[m,:,:,:] = w[m]*P/nϕ
        end

        Jμλ = sum(Jμ, dims=1)[1,:,:,:] .* 0.5

        h5open(output_path, "r+") do file
            file["J"][iteration, λi,:,:,:] = ustrip.(Jμλ)
            file["time"][iteration, λi] = et
        end
    end

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
