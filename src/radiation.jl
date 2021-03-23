include("atmosphere.jl")
include("rates.jl")
include("atom.jl")

struct Radiation
    α_continuum::Array{<:PerLength, 4}             # (nλ, nz, nx, ny)
    ε_continuum::Array{Float64,4}                  # (nλ, nz, nx, ny)
    α_line_constant::Array{Float64, 3}             # (nz, nx, ny)
    ε_line::Array{Float64,3}                       # (nz, nx, ny)
    B::Array{<:UnitsIntensity_λ,4}                 # (nλ, nz, nx, ny)
end

struct RadiationBackground
    λ::Array{<:Unitful.Length, 1}                  # (nλ)
    α_continuum::Array{<:PerLength, 4}             # (nλ, nz, nx, ny)
    ε_continuum::Array{Float64,4}                  # (nλ, nz, nx, ny)
    B::Array{<:UnitsIntensity_λ,4}                 # (nλ, nz, nx, ny)
end

"""
    collect_radiation_data(atmosphere::Atmosphere,
                           λ::Unitful.Length)

Collects radition data for background processes at a single wavelength
Returns data to go into structure.
"""
function collect_radiation_data(atmosphere::Atmosphere,
                                λ::Unitful.Length)

    # ==================================================================
    # GET ATMOSPHERE DATA AND WAVELENGTH
    # ==================================================================
    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    hydrogen_populations = atmosphere.hydrogen_populations
    nz, nx, ny = size(temperature)

    λ = [λ]
    # ==================================================================
    # INITIALISE VARIABLES
    # ==================================================================
    α = Array{PerLength,4}(undef, 1, nz, nx, ny)
    ε = Array{Float64,4}(undef, 1, nz, nx, ny)
    B = Array{Float64,4}(undef, 1, nz, nx, ny)*u"kW / m^2 / sr / nm"

    # ==================================================================
    # EXTINCTION AND DESTRUCTION PROBABILITY FOR BACKGROUND PROCESSES
    # ==================================================================
    proton_density = hydrogen_populations[:,:,:,end]
    hydrogen_ground_density = hydrogen_populations[:,:,:,1]
    hydrogen_neutral_density = hydrogen_populations[:,:,:,1] .+ hydrogen_populations[:,:,:,2]

    α_continuum_abs = α_cont_abs.(λ, temperature, electron_density, hydrogen_neutral_density, proton_density)
    α_continuum_scat = α_cont_scatt.(λ, electron_density, hydrogen_ground_density)

    α[1,:,:,:] = α_continuum_abs .+ α_continuum_scat
    ε[1,:,:,:] = α_continuum_abs ./ α[1,:,:,:]

    # ==================================================================
    # PLANCK FUNCTION
    # ==================================================================
    B[1,:,:,:] = blackbody_lambda.(λ[1], temperature)

    # ==================================================================
    # CHECK FOR UNVALID VALUES
    # ==================================================================
    @test all(  Inf .> ustrip.(λ) .>= 0.0 )
    @test all(  Inf .> ustrip.(B) .>= 0.0 )
    @test all(  Inf .> ustrip.(α) .>= 0.0 )
    @test all(  1.0 .>= ε .>= 0.0)

    return λ, α, ε, B
end

"""
    collect_radiation_data(atmosphere::Atmosphere,
                           atom::Atom,
                           rates::TransitionRates,
                           populations::Array{<:NumberDensity,4})

Collects radition data wavelength associated with bound-bound and
bound-free processes. Returns data to go into structure.
"""
function collect_radiation_data(atmosphere::Atmosphere,
                                atom::Atom,
                                rates::TransitionRates,
                                populations::Array{<:NumberDensity,4})

    # ==================================================================
    # GET ATOM DATA
    # ==================================================================
    line = atom.line
    λ = atom.λ

    # ==================================================================
    # EXTINCTION AND DESTRUCTION PROBABILITY FOR EACH WAVELENGTH
    # ==================================================================
    α_continuum, ε_continuum = continuum_extinction_destruction(atmosphere, atom, rates, populations, λ)
    ε_line = line_destruction(rates)
    α_line_constant = line_extinction_constant.(Ref(line), populations[:,:,:,1], populations[:,:,:,2])

    # ==================================================================
    # PLANCK FUNCTION
    # ==================================================================
    B = blackbody_lambda(λ, atmosphere.temperature)

    # ==================================================================
    # CHECK FOR UNVALID VALUES
    # ==================================================================
    @test all( Inf .> ustrip.(α_continuum) .>= 0.0 )
    @test all( 1.0 .>= ε_continuum .>= 0.0 )
    @test all( 1.0 .>= ε_line .>= 0.0 )
    @test all( Inf .> ustrip.(α_line_constant) .>= 0.0 )
    @test all( Inf .> ustrip.(B) .>= 0.0 )

    return α_continuum, ε_continuum, α_line_constant, ε_line, B
end

# ==================================================================
# EXTINCTION AND DESTRUCTION
# ==================================================================

"""
    continuum_extinction_destruction(atmosphere::Atmosphere,
                                     atom::Atom,
                                     rates::TransitionRates,
                                     atom_populations::Array{<:NumberDensity,4},
                                     λ::Array{<:Unitful.Length, 1})

Collect non-line extinction and destruction for all wavelengths.
Includes H bf, H- bf and ff, H2+ bf and ff, thomson and rayleigh.
"""

function continuum_extinction_destruction(atmosphere::Atmosphere,
                                          atom::Atom,
                                          rates::TransitionRates,
                                          atom_populations::Array{<:NumberDensity,4},
                                          λ::Array{<:Unitful.Length, 1})

    # ==================================================================
    # EXTINCTION AND DESTRUCTION PROBABILITY FROM BACKGROUND PROCESSES
    # ==================================================================

    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    hydrogen_populations = atmosphere.hydrogen_populations

    nλ_bf = atom.nλ_bf
    nλ_bb = atom.nλ_bb

    nz, nx, ny = size(temperature)
    nλ = length(λ)
    λ0 = λ[nλ_bf*2 + (nλ_bb÷2) + 1]

    # ==================================================================
    # EXTINCTION AND DESTRUCTION PROBABILITY FROM BACKGROUND PROCESSES
    # ==================================================================
    α_background = Array{PerLength, 4}(undef, 2nλ_bf, nz, nx, ny)
    ε_background = Array{Float64,4}(undef, 2nλ_bf, nz, nx, ny)
    α_abs = Array{PerLength, 4}(undef, 2nλ_bf, nz, nx, ny)
    α_scatt = Array{PerLength, 4}(undef, 2nλ_bf, nz, nx, ny)
    α_continuum = Array{PerLength, 4}(undef, nλ, nz, nx, ny)
    ε_continuum = Array{Float64,4}(undef, nλ, nz, nx, ny)

    proton_density = hydrogen_populations[:,:,:,end]
    hydrogen_ground_density = hydrogen_populations[:,:,:,1]
    hydrogen_neutral_density = hydrogen_populations[:,:,:,1] .+ hydrogen_populations[:,:,:,2]

    # Background at bound-free wavelengths
    @Threads.threads for l=1:2*nλ_bf
        α_abs[l,:,:,:] = α_cont_abs.(λ[l], temperature, electron_density, hydrogen_neutral_density, proton_density)
        α_scatt[l,:,:,:] = α_cont_scatt.(λ[l], electron_density, hydrogen_ground_density)

        α_background[l,:,:,:] = α_scatt[l,:,:,:] .+ α_abs[l,:,:,:]
        ε_background[l,:,:,:] = α_abs[l,:,:,:] ./ α_background[l,:,:,:]
    end

    @test all(1.0  .>= ε_background .>= 0.0)

    # ==================================================================
    # BACKGROUND EXTINCTION AND DESTRUCTION FOR LINE
    # ==================================================================

    # Assume constant background over line profile wavelengths
    α_abs = α_cont_abs.(λ0, temperature, electron_density, hydrogen_neutral_density, proton_density)
    α_scatt =  α_cont_scatt.(λ0, electron_density, hydrogen_ground_density)
    α_background_line = α_abs .+ α_scatt
    ε_background_line = α_abs ./ α_background_line

    @Threads.threads for l=2*nλ_bf+1:nλ
        α_continuum[l,:,:,:] = α_background_line
        ε_continuum[l,:,:,:] = ε_background_line
    end

    # ==================================================================
    # EXTINCTION AND DESTRUCTION FROM ATOM BOUND-FREE
    # ==================================================================
    ν = c_0 ./ λ
    n_eff = sqrt(E_∞ / (atom.χu - atom.χl)) |> u"J/J"

    C31 = rates.C31
    R31 = rates.R31
    C32 = rates.C32
    R32 = rates.R32

    ε_bf_l = C31 ./ (R31 .+ C31)
    ε_bf_u = C32 ./ (R32 .+ C32)

    @Threads.threads for l=1:nλ_bf
        α_bf_l = hydrogenic_bf.(ν[l], ν[nλ_bf],
                               temperature,  atom_populations[:,:,:,1],
                               1.0, n_eff)

        α_bf_u = hydrogenic_bf.(ν[l+nλ_bf], ν[2*nλ_bf],
                               temperature, atom_populations[:,:,:,2],
                               1.0, n_eff)

        α_continuum[l,:,:,:] = α_background[l,:,:,:] .+ α_bf_l
        ε_continuum[l,:,:,:] = ( ε_background[l,:,:,:] .* α_background[l,:,:,:] .+ ε_bf_l .* α_bf_l ) ./ α_continuum[l,:,:,:]

        α_continuum[l+nλ_bf,:,:,:] = α_background[l+nλ_bf,:,:,:] .+ α_bf_u
        ε_continuum[l+nλ_bf,:,:,:] = ( ε_background[l+nλ_bf,:,:,:] .* α_background[l+nλ_bf,:,:,:] .+ ε_bf_u .* α_bf_u ) ./ α_continuum[l+nλ_bf,:,:,:]
    end

    return α_continuum, ε_continuum
end

"""
    line_extinction(λ::Unitful.Length,
                    λ0::Unitful.Length,
                    ΔλD::Unitful.Length,
                    damping_constant::PerArea,
                    α_line_constant::Float64,
                    v_los::Unitful.Velocity=0u"m/s")

Calculate line profile and return bound-bound
extinction contribution for a line wavelength.
"""
function line_extinction(λ::Unitful.Length,
                         λ0::Unitful.Length,
                         ΔλD::Unitful.Length,
                         damping_constant::PerArea,
                         α_line_constant::Float64,
                         v_los::Unitful.Velocity=0u"m/s")

    damping = damping_constant*λ^2 |> u"m/m"
    v = (λ - λ0 .+ λ0 .* v_los ./ c_0) ./ ΔλD
    profile = voigt_profile.(damping, ustrip(v), ΔλD)
    α = α_line_constant * profile

    @test all(  Inf .> ustrip.(α) .>= 0.0 )

    return α
end


"""
    line_destruction(rates::TransitionRates)

Returns line destruction probability for the two level atom.
"""
function line_destruction(rates::TransitionRates)
    C21 = rates.C21
    R21 = rates.R21
    return C21 ./ (R21 .+ C21)
end


"""
    line_extinction_constant(line::AtomicLine, n_l::NumberDensity, n_u::NumberDensity)

Compute the line extinction constant to be
multiplied by the profile (per length).
"""
function line_extinction_constant(line::AtomicLine, n_l::NumberDensity, n_u::NumberDensity)
    (h * c_0 / (4 * π * line.λ0) * (n_l * line.Bij - n_u * line.Bji)) |> u"m/m"
end

"""
    α_cont_abs(λ::Unitful.Length,
               temperature::Unitful.Temperature,
               electron_density::NumberDensity,
               h_neutral_density::NumberDensity,
               proton_density::NumberDensity)

The extinction from continuum absorption processes for a given λ.
Includes H- ff, H- bf, H ff, H2+ ff and H2+ bf. Credit: Tiago
"""
function α_cont_abs(λ::Unitful.Length,
                    temperature::Unitful.Temperature,
                    electron_density::NumberDensity,
                    h_neutral_density::NumberDensity,
                    proton_density::NumberDensity)

    α = Transparency.hminus_ff_stilley(λ, temperature, h_neutral_density, electron_density)
    α += Transparency.hminus_bf_geltman(λ, temperature, h_neutral_density, electron_density)
    α += hydrogenic_ff(c_0 / λ, temperature, electron_density, proton_density, 1)
    α += h2plus_ff(λ, temperature, h_neutral_density, proton_density)
    α += h2plus_bf(λ, temperature, h_neutral_density, proton_density)
    return α
end

"""
    α_cont_scatt(λ::Unitful.Length,
                 electron_density::NumberDensity,
                 h_ground_density::NumberDensity)

The extincion from Thomson and Rayleigh scattering
for a given λ. Credit: Tiago
"""
function α_cont_scatt(λ::Unitful.Length,
                      electron_density::NumberDensity,
                      h_ground_density::NumberDensity)

    α = thomson(electron_density)
    α += rayleigh_h(λ, h_ground_density)
    return α
end



"""
    optical_depth(α::Array{<:PerLength, 3},
                  z::Array{<:Unitful.Length, 1})

Calculates the vertical optical depth of the atmosphere.
"""
function optical_depth(α::Array{<:PerLength, 3},
                       z::Array{<:Unitful.Length, 1},
                       μ::Float64)

    nz, nx, ny = size(α)
    τ = Array{Float64,3}(undef, nz-1, nx, ny)

    # Calculate vertical optical depth for each column
    Threads.@threads for col=1:nx*ny
        j = 1 + (col-1)÷nx
        i = col - (j-1)*nx
        τ[1,i,j] = 0.5(z[1] - z[2]) * (α[1,i,j] + α[2,i,j])

        for k=2:nz-1
            τ[k,i,j] =  τ[k-1,i,j] + 0.5(z[k] - z[k+1]) * (α[k,i,j] + α[k+1,i,j])
        end
    end

    return τ ./ μ
end

"""
    blackbody_lambda(λ::Unitful.Length,
                     temperature::Unitful.Temperature)

Calculates the Blackbody (Planck) function per
wavelength, for a given wavelength and temperature.
Returns monochromatic intensity. Credit: Tiago
"""
function blackbody_lambda(λ::Unitful.Length,
                          temperature::Unitful.Temperature)
    B = (2h * c_0^2) / ( λ^5 * (exp((h * c_0 / k_B) / (λ * temperature)) - 1) ) |> u"kW / m^2 / sr / nm"
end

"""
    blackbody_lambda(λ::Array{<:Unitful.Length,1},
                     temperature::Unitful.Temperature)

Calculates the Blackbody (Planck) function per
wavelength, for an array of wavelengths and 3D temperature.
Returns monochromatic intensity.
"""
function blackbody_lambda(λ::Array{<:Unitful.Length,1},
                          temperature::Array{<:Unitful.Temperature,3})
    nλ = length(λ)
    nz, nx, ny = size(temperature)
    B = Array{UnitsIntensity_λ, 4}(undef, nλ, nz, nx, ny)

    for l=1:nλ
        B[l,:,:,:] = (2h * c_0^2) ./ ( λ[l]^5 * (exp.((h * c_0 / k_B) ./ (λ[l] * temperature)) .- 1) ) .|> u"kW / m^2 / sr / nm"
    end

    return B
end
