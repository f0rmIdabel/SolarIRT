include("atmosphere.jl")
include("rates.jl")
include("atom.jl")

struct Radiation
    α_continuum::Array{<:PerLength, 4}                     # (nλ, nz, nx, ny)
    α_line_constant::Array{Float64, 3}
    B::Array{<:UnitsIntensity_λ,4}
end

struct RadiationBackground
    λ::Array{<:Unitful.Length, 1}                          # (nλ)
    α_continuum::Array{<:PerLength, 4}                     # (nλ, nz, nx, ny)
    B::Array{<:UnitsIntensity_λ,4}
end

"""
TEST MODE: BACKGROUND PROCESSES
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
    # EXTINCTION FOR BACKGROUND PROCESSES
    # ==================================================================
    α = Array{PerLength,4}(undef, 1, nz, nx, ny)
    proton_density = hydrogen_populations[:,:,:,end]
    hydrogen_ground_popuplation = hydrogen_populations[:,:,:,1]

    α_abs = α_cont_abs.(λ, temperature, electron_density, hydrogen_ground_popuplation, proton_density)
    α_scat = α_cont_scatt.(λ, electron_density, hydrogen_ground_popuplation)
    α[1,:,:,:] = α_abs .+ α_scat

    # ==================================================================
    # PLANCK FUNCTION
    # ==================================================================
    Bλ[1,:,:,:] = blackbody_lambda.(λ[1], temperature)

    return λ, α, B
end

"""
FULL MODE: POPULATION ITERATION
Collects radition data wavelength associated with bound-bound and bound-free processes.
Returns data to go into structure.
"""
function collect_radiation_data(atmosphere::Atmosphere,
                                atom::Atom,
                                populations::Array{<:NumberDensity,4})

    # ==================================================================
    # GET ATMOSPHERE DATA
    # ==================================================================
    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    hydrogen_populations = atmosphere.hydrogen_populations
    nz, nx, ny = size(temperature)
    velocity_z = atmosphere.velocity_z

    # ==================================================================
    # GET ATOM DATA
    # ==================================================================
    populations = populations
    line = atom.line
    λ0 = line.λ0
    ΔλD = atom.doppler_width
    damping_constant = atom.damping_constant
    nλ_bf = atom.nλ_bf
    λ = atom.λ
    nλ = length(λ)

    # ==================================================================
    # INITIALISE VARIABLES
    # ==================================================================
    boundary = Array{Int32,3}(undef, nλ, nx, ny)
    packets = Array{Int32,4}(undef, nλ, nz, nx, ny)
    intensity_per_packet =  Array{UnitsIntensity_λ, 1}(undef, nλ)

    # ==================================================================
    # EXTINCTION AND DESTRUCTION PROBABILITY FOR EACH WAVELENGTH
    # ==================================================================
    α_continuum = continuum_extinction(atmosphere, atom, populations, λ)
    α_line_constant = line_extinction_constant.(Ref(line), populations[:,:,:,1], populations[:,:,:,2])

    return α_continuum, α_line_constant
end

function continuum_extinction(atmosphere::Atmosphere,
                              atom::Atom,
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
    α_continuum = Array{PerLength, 4}(undef, nλ, nz, nx, ny)

    proton_density = hydrogen_populations[:,:,:,end]
    hydrogen_ground_popuplation = hydrogen_populations[:,:,:,1]

    # Background at bound-free wavelengths
    @Threads.threads for l=1:2*nλ_bf
        α_abs = α_cont_abs.(λ[l], temperature, electron_density, hydrogen_ground_popuplation, proton_density)
        α_scatt = α_cont_scatt.(λ[l], electron_density, hydrogen_ground_popuplation)
        α_background[l,:,:,:] = α_scatt .+ α_abs
    end

    # ==================================================================
    # BACKGROUND EXTINCTION AND DESTRUCTION FOR LINE
    # ==================================================================

    # Assume constant background over line profile wavelengths
    α_abs = α_cont_abs.(λ0, temperature, electron_density, hydrogen_ground_popuplation, proton_density)
    α_scatt =  α_cont_scatt.(λ0, electron_density, hydrogen_ground_popuplation)
    α_background_line = α_abs .+ α_scatt

    @Threads.threads for l=2*nλ_bf+1:nλ
        α_continuum[l,:,:,:] = α_background_line
    end

    # ==================================================================
    # EXTINCTION AND DESTRUCTION FROM ATOM BOUND-FREE
    # ==================================================================
    ν = c_0 ./ λ
    n_eff = sqrt(E_∞ / (atom.χl - atom.χu)) |> u"J/J"

    @Threads.threads for l=1:nλ_bf
        α_bf_l = hydrogenic_bf.(ν[l], ν[nλ_bf],
                               temperature,  atom_populations[:,:,:,1],
                               1.0, n_eff)

        α_bf_u = hydrogenic_bf.(ν[l+nλ_bf], ν[2*nλ_bf],
                               temperature, atom_populations[:,:,:,2],
                               1.0, n_eff)

        α_continuum[l,:,:,:] = α_background[l,:,:,:] .+ α_bf_l
        α_continuum[l+nλ_bf,:,:,:] = α_background[l+nλ_bf,:,:,:] .+ α_bf_u
    end

    return α_continuum
end

function line_extinction(λ::Unitful.Length,
                         λ0::Unitful.Length,
                         ΔλD::Unitful.Length,
                         damping_constant::PerArea,
                         α_line_constant::Float64,
                         v_los::Unitful.Velocity=0.0u"m/s")

    damping = damping_constant*λ^2 |> u"m/m"
    v = (λ - λ0 .+ λ0 .* v_los ./ c_0) ./ ΔλD
    profile = voigt_profile.(damping, ustrip(v), ΔλD)
    α = α_line_constant * profile

    return α
end

"""
Compute line extinction given an `AtomicLine` struct, `profile` defined per wavelength,
and upper and lower population densities `n_u` and `n_l`.
"""
function line_extinction_constant(line::AtomicLine, n_u::NumberDensity, n_l::NumberDensity)
    (h * c_0 / (4 * π * line.λ0) * (n_l * line.Bij - n_u * line.Bji)) |> u"m/m"
end

"""
The extinction from continuum absorption processes for a given λ.
Includes H- ff, H- bf, H ff, H2+ ff and H2+ bf.
Credit: Tiago
"""
function α_cont_abs(λ::Unitful.Length,
                    temperature::Unitful.Temperature,
                    electron_density::NumberDensity,
                    h_ground_density::NumberDensity,
                    proton_density::NumberDensity)

    α = Transparency.hminus_ff_stilley(λ, temperature, h_ground_density, electron_density)
    α += Transparency.hminus_bf_geltman(λ, temperature, h_ground_density, electron_density)
    α += hydrogenic_ff(c_0 / λ, temperature, electron_density, proton_density, 1)
    α += h2plus_ff(λ, temperature, h_ground_density, proton_density)
    α += h2plus_bf(λ, temperature, h_ground_density, proton_density)
    return α
end

"""
The extincion from Thomson and Rayleigh scattering for a given λ.
Credit: Tiago
"""
function α_cont_scatt(λ::Unitful.Length,
                      electron_density::NumberDensity,
                      h_ground_density::NumberDensity)

    α = thomson(electron_density)
    α += rayleigh_h(λ, h_ground_density)
    return α
end

"""
Calculates the vertical optical depth of the atmosphere.
"""
function optical_depth(α::Array{<:PerLength, 3},
                       z::Array{<:Unitful.Length, 1})

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

    return τ
end

"""
Calculates the Blackbody (Planck) function per wavelength,
for given arrays of wavelength and temperature.
Returns monochromatic intensity.
"""
function blackbody_lambda(λ::Unitful.Length,
                          temperature::Unitful.Temperature)
    B = (2h * c_0^2) / ( λ^5 * (exp((h * c_0 / k_B) / (λ * temperature)) - 1) ) |> u"kW / m^2 / sr / nm"
end

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

function transition_λ(χ1::Unitful.Energy, χ2::Unitful.Energy)
    ((h * c_0) / (χ2-χ1)) |> u"nm"
end

function write_to_file(radiation::RadiationBackground)
    h5open("../out/output.h5", "w") do file
        write(file, "extinction_continuum", ustrip(radiation.α_continuum))
    end
end

function write_to_file(radiation::Radiation)
    h5open("../out/output.h5", "w") do file
        write(file, "extinction_continuum", ustrip(radiation.α_continuum))
        write(file, "extinction_line_constant", radiation.α_line_constant)
    end
end
