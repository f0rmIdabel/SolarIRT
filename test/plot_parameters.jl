using Plots
using Unitful
import Statistics


"""
    plot_atmosphere(atmosphere::Atmosphere)

Plot column averaged temperature, electron density,
hydrogen populations and speed for an atmosphere.
"""
function plot_atmosphere(atmosphere::Atmosphere)
    # ===========================================================
    # LOAD DATA
    # ===========================================================
    z = atmosphere.z[1:end-1]
    T = atmosphere.temperature
    electron_density = atmosphere.electron_density
    hydrogen_populations = atmosphere.hydrogen_populations
    v = atmosphere.velocity
    speed = velocity_to_speed(v)

    # ===========================================================
    # GET AVERAGE COLUMN
    # ===========================================================

    mean_T = average_column(ustrip.(T))
    mean_electron_density = average_column(ustrip.(electron_density))
    mean_speed = average_column(ustrip.(speed))
    hydrogen_populations = ustrip.(hydrogen_populations)
    mean_h1 = average_column(hydrogen_populations[:,:,:,1])
    mean_h2 = average_column(hydrogen_populations[:,:,:,2])
    mean_h3 = average_column(hydrogen_populations[:,:,:,3])
    total =  mean_h1 .+ mean_h2 .+ mean_h3

    # ===========================================================
    # PLOT
    # ===========================================================

    ENV["GKSwstype"]="nul"
    z = ustrip.(z .|>u"Mm")

    p1 = Plots.plot(z, ustrip.(mean_T),
                    xlabel = "z (Mm)", ylabel = "temperature (K)",
                    yscale=:log10, legend = false)
    p2 = Plots.plot(z, ustrip.(mean_speed),
                    xlabel = "z (Mm)", ylabel = "speed (m/s)",
                    legend = false)
    p3 = Plots.plot(z, ustrip.(mean_electron_density),
                    xlabel = "z (Mm)", ylabel = "electron density (m^-3)",
                    yscale=:log10, legend = false)
    p4 = Plots.plot(z, [mean_h1./total, mean_h2./total, mean_h3./total],
                    xlabel = "z (Mm)", ylabel = "hydrogen density (m^-3)",
                    yscale=:log10,
                    label=permutedims(["ground","excited","ionised"]),
                    legendfontsize=6)

    Plots.plot(p1, p2, p3, p4, layout = (2, 2))
    Plots.png("plots/atmosphere")
end

"""
    plot_populations(populations::Array{<:NumberDensity, 3},
                     z::Array{<:Unitful.Length,1})

Plot the population distribution for a 2-level atom with continuum.
"""
function plot_populations(populations_LTE::Array{<:NumberDensity, 4},
                          populations_zero_radiation::Array{<:NumberDensity, 4},
                          z::Array{<:Unitful.Length,1})
    # ===========================================================
    # GET AVERAGE COLUMN
    # ===========================================================
    mean_p1 = average_column(populations_LTE[:,:,:,1])
    mean_p2 = average_column(populations_LTE[:,:,:,2])
    mean_p3 = average_column(populations_LTE[:,:,:,3])
    total =  mean_p1 .+ mean_p2 .+ mean_p3

    mean_p1_z = average_column(populations_zero_radiation[:,:,:,1])
    mean_p2_z = average_column(populations_zero_radiation[:,:,:,2])
    mean_p3_z = average_column(populations_zero_radiation[:,:,:,3])
    total_z =  mean_p1_z .+ mean_p2_z .+ mean_p3_z

    # ===========================================================
    # PLOT
    # ===========================================================
    ENV["GKSwstype"]="nul"
    z = ustrip.(z .|>u"Mm")

    p1 = Plots.plot(z, [mean_p1./total, mean_p2./total, mean_p3./total],
                    xlabel = "z (Mm)", ylabel = "population density (m^-3)",
                    yscale=:log10, label=permutedims(["ground","excited","ionised"]))
    p2 = Plots.plot(z, [mean_p1_z./total_z, mean_p2_z./total_z, mean_p3_z./total_z],
                    xlabel = "z (Mm)", ylabel = "population density (m^-3)",
                    yscale=:log10, label=permutedims(["ground","excited","ionised"]))
    Plots.plot(p1, p2, title=permutedims(["LTE", "Zero radiation"]), layout=(2,1))
    Plots.png("plots/initial_populations")
end

"""
    plot_radiationBackground(radiationBackground::RadiationBackground,
                             z::Array{<:Unitful.Length, 1})

Plots column averaged extincion and destruction probability.
"""
function plot_radiationBackground(radiationBackground::RadiationBackground,
                                  z::Array{<:Unitful.Length, 1})
    #λ = radiationBackground.λ
    α_continuum = radiationBackground.α_continuum[1,:,:,:]
    ε_continuum = radiationBackground.ε_continuum[1,:,:,:]

    mean_α = average_column(ustrip.(α_continuum))u"m^-1"
    mean_ε = average_column(ε_continuum)


    ENV["GKSwstype"]="nul"
    z = ustrip.(z .|>u"Mm")
    mean_α = ustrip.(mean_α)

    p1 = Plots.plot(z, mean_α,
                    xlabel = "z (Mm)", ylabel = "Extinction (m^-1)",
                    yscale=:log10, legend = false)
    p2 = Plots.plot(z, mean_ε,
                    xlabel = "z (Mm)", ylabel = "Destruction",
                    yscale=:log10)
    Plots.plot(p1, p2, legend=false)
    Plots.png("plots/radiation_background")
end


"""
    plot_rates(rates::TransitionRates,
               z::Array{<:Unitful.Length, 1})

Plot the column averaged transition rates.
"""
function plot_rates(rates::TransitionRates,
                    z::Array{<:Unitful.Length, 1})

    R12 = average_column(ustrip.(rates.R12))
    R13 = average_column(ustrip.(rates.R13))
    R23 = average_column(ustrip.(rates.R23))
    R21 = average_column(ustrip.(rates.R21))
    R31 = average_column(ustrip.(rates.R31))
    R32 = average_column(ustrip.(rates.R32))
    C12 = average_column(ustrip.(rates.C12))
    C13 = average_column(ustrip.(rates.C13))
    C23 = average_column(ustrip.(rates.C23))
    C21 = average_column(ustrip.(rates.C21))
    C31 = average_column(ustrip.(rates.C31))
    C32 = average_column(ustrip.(rates.C32))

    ENV["GKSwstype"]="nul"
    z = ustrip.(z .|>u"Mm")
    p1 = Plots.plot(z, [R12, C12],
                    ylabel = "rates (s^-1)", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R12","C12"]),
                    legendfontsize=6)

    p2 = Plots.plot(z, [R13, C13],
                    ylabel = "rates (s^-1)", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R13","C13"]),
                    legendfontsize=6)

    p3 = Plots.plot(z, [R23, C23],
                    ylabel = "rates [s^-1]", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R23","C23"]),
                    legendfontsize=6)

    p4 = Plots.plot(z, [ R21, C21],
                    ylabel = "rates (s^-1)", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R21","C21"]),
                    legendfontsize=6)

    p5 = Plots.plot(z, [R31, C31],
                    ylabel = "rates (s^-1)", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R31","C31"]),
                    legendfontsize=6)

    p6 = Plots.plot(z, [R32, C32],
                    ylabel = "rates (s^-1)", xlabel = "z (Mm)",
                    yscale=:log10, label=permutedims(["R32","C32"]),
                    legendfontsize=6)

    Plots.plot(p1, p2, p3, p4, p5, p6, tickfontsize=6)
    Plots.png("plots/transition_rates")
end


"""
    plot_radiation(radiation::Radiation,
                   atom::Atom,
                   z::Array{<:Unitful.Length, 1})

For bb-center and bf-edge wavelengths, plot average column extinction
and average column destruction probability.
"""
function plot_radiation(radiation::Radiation,
                        atom::Atom,
                        z::Array{<:Unitful.Length, 1})

    # ===========================================================
    # LOAD RADIATION DATA
    # ===========================================================
    α_continuum = radiation.α_continuum
    ε_continuum = radiation.ε_continuum
    α_line_constant = radiation.α_line_constant
    ε_line = radiation.ε_line

    nλ, nz, nx, ny = size(α_continuum)

    # ===========================================================
    # LOAD ATOM DATA AND GET LINE OPACITY/DESTRUCTION
    # ===========================================================
    λ = atom.λ
    nλ_bb = atom.nλ_bb
    nλ_bf = atom.nλ_bf

    α_total = copy(α_continuum)
    ε_total = copy(ε_continuum)

    for l=1:nλ_bb
        α_line = line_extinction.(λ[2nλ_bf + l], atom.line.λ0, atom.doppler_width, atom.damping_constant, α_line_constant)

        α_total[(2nλ_bf + l),:,:,:] += α_line
        ε_total[(2nλ_bf + l),:,:,:] = (ε_continuum[(2nλ_bf + l),:,:,:].*α_continuum[(2nλ_bf + l),:,:,:]  .+  ε_line.*α_line) ./ α_total[(2nλ_bf + l),:,:,:]
    end

    α_total = ustrip.(α_total)

    c = 2nλ_bf + nλ_bb ÷ 2 + 1
    mean_α_bb = average_column(α_total[c,:,:,:])
    mean_α_bf_l = average_column(α_total[nλ_bf,:,:,:])
    mean_α_bf_u = average_column(α_total[2nλ_bf,:,:,:])

    mean_ε_bb = average_column(ε_total[c,:,:,:])
    mean_ε_bf_l = average_column(ε_total[nλ_bb,:,:,:])
    mean_ε_bf_u = average_column(ε_total[2nλ_bf,:,:,:])

    z = ustrip.(z .|>u"Mm")
    λ = ustrip.(λ .|>u"nm")

    ENV["GKSwstype"]="nul"
    p1 = Plots.plot(z, [mean_α_bb, mean_α_bf_l, mean_α_bf_u ],
                    ylabel = "Extinction (m^-1)", xlabel = "z (Mm)",
                    yscale=:log10,
                    label=permutedims(["1 -> 2", "1 -> c", "2 -> c"]))
    p2 = Plots.plot(z, [mean_ε_bb, mean_ε_bf_l, mean_ε_bf_u ],
                    ylabel = "Destruction", xlabel = "z (Mm)",
                    yscale=:log10,
                    label=permutedims(["1 -> 2", "1 -> c", "2 -> c"]))

    Plots.plot(p1, p2, tickfontsize=6, legendfontsize=6, layout=(2,1))
    Plots.png("plots/radiation_atmosphere")
end

"""
    average_column(array::Array{Real,3})

Given a (nz, nx, ny)-dimensional array,
get the (nz)-dimensional average column.
"""
function average_column(array::Array)
      Statistics.mean(array, dims=[2,3])[:,1,1]
end


"""
    velocity_to_speed(velocity::Array{Array{<:Unitful.Velocity, 1}, 3})

Given a (nz, nx, ny)-dimensional array containing
3-dimensional velocity [vz,vx,vy] arrays,
calculate the (nz, nx,ny)-dimensional speed.
"""
function velocity_to_speed(velocity::Array{Array{<:Unitful.Velocity, 1}, 3})
    nz, nx, ny = size(velocity)
    speed = Array{Float64, 3}(undef, nz,nx,ny)

    for j=1:ny
        for i=1:nx
            for k=1:nz
                speed[k,i,j] = ustrip(sqrt(velocity[k,i,j][1]^2 + velocity[k,i,j][2]^2 + velocity[k,i,j][3]^2))
            end
        end
    end

    return speed*u"m/s"
end
