include("lambda_iteration.jl")

function run()
    println("\n", "="^83, "\n", " "^30,
            "SOLAR ATMOSPHERE MCRT",
            "\n", "="^83)

    # ==================================================================
    # LOAD ATMOSPHERE DATA
    # ==================================================================
    print("--Loading atmosphere data..................")
    atmosphere_parameters = collect_atmosphere_data()
    atmosphere = Atmosphere(atmosphere_parameters...)
    println("Atmosphere loaded with dimensions ", size(atmosphere.temperature), ".")

    mode = get_mode()

    if mode == "test"

        # ==================================================================
        # LOAD WAVELENGTH
        # ==================================================================
        print("--Loading wavelength.......................")
        λ = get_test_λ()
        println("Wavelength λ = ", λ, " loaded.")

        # ==================================================================
        # LOAD RADIATION DATA
        # ==================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_radiation_data(atmosphere, λ)
        radiation = Radiation(radiation_parameters...)
        println(@sprintf("Radiation loaded with %.2e packets.", sum(radiation.packets)))

        # ==================================================================
        # FEAUTRIER CALCULATION
        # ==================================================================
        lambda_iteration(atmosphere, radiation)
        println(" λ-iteration finished.")

    elseif mode == "atom"

        # ==================================================================
        # LOAD ATOM
        # ==================================================================
        atom_parameters = collect_atom_data()
        atom = AtomicLine(collect_atom_data()...)

        # ==================================================================
        # LOAD INITIAL ATOM POPULATIONS
        # ==================================================================
        populations = collect_initial_populations(atmosphere.hydrogen_populations)

        # ==================================================================
        # LOAD RADIATION DATA
        # ==================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_radiation_data(atmosphere, atom, populations)
        radiation = Radiation(radiation_parameters...)

        # ==================================================================
        # FEAUTRIER CALCULATION
        # ==================================================================
        lambda_iteration(atmosphere, radiation)
    end
end

include("../src/mcrt.jl")
include("../src/populations.jl")

function run()
    println("\n", "="^91, "\n", " "^34,
            "SOLAR ATMOSPHERE RT",
            "\n", "="^91, "\n")

    # =============================================================================
    # LOAD ATMOSPHERE DATA
    # =============================================================================
    print("--Loading atmosphere data..................")
    atmosphere_parameters = collect_atmosphere_data()
    atmosphere = Atmosphere(atmosphere_parameters...)
    println("Atmosphere loaded with dimensions ", size(atmosphere.temperature), ".")

    if test_mode()
        # =============================================================================
        # LOAD WAVELENGTH
        # =============================================================================
        print("--Loading wavelength.......................")
        λ = get_background_λ()
        println("Wavelength λ = ", λ, " loaded.")

        # =============================================================================
        # LOAD RADIATION DATA
        # =============================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_radiation_data(atmosphere, λ)
        radiation = RadiationBackground(radiation_parameters...)
        write_to_file(radiation) # creates new file
        println(@sprintf("Radiation loaded with %.2e packets.", sum(radiation.packets)))

        # =============================================================================
        # SIMULATION
        # =============================================================================
        mcrt(atmosphere, radiation)

    else
        # =============================================================================
        # LOAD ATOM
        # =============================================================================
        print("--Loading atom.............................")
        atom_parameters = collect_atom_data(atmosphere)
        atom = Atom(atom_parameters...)
        println("Atom loaded with ", atom.nλ_bb + 2*atom.nλ_bf, " wavelengths.")

        # =============================================================================
        # LOAD INITIAL POPULATIONS
        # =============================================================================
        print("--Loading initial populations..............")
        populations = collect_initial_populations()
        println("Initial populations loaded.")

        # =============================================================================
        # CALCULATE INITIAL TRANSITION RATES
        # =============================================================================
        print("--Loading initial transition rates.........")
        Bλ = blackbody_lambda(atom.λ, atmosphere.temperature)
        rate_parameters = calculate_transition_rates(atom, atmosphere, populations, Bλ)
        rates = TransitionRates(rate_parameters...)
        println("Initial transition rates loaded.")

        # =============================================================================
        # RUN MCRT UNTIL POPULATIONS CONVERGE
        # =============================================================================
        converged_populations = false
        max_iterations = get_max_iterations()

        # =============================================================================
        # LOAD RADIATION DATA WITH CURRENT POPULATIONS
        # =============================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_radiation_data(atmosphere, atom, rates, populations)
        radiation = Radiation(radiation_parameters...)
        write_to_file(radiation) # creates new file
        write_to_file(atom.λ)
        println(@sprintf("Radiation loaded with %.2e packets per λ.", sum(radiation.packets[1,:,:,:])))

        # ==================================================================
        # FEAUTRIER CALCULATION
        # ==================================================================
        lambda_iteration(atmosphere, radiation)

        # =============================================================================
        # END OF ATOM MODE
        # =============================================================================
    end

end

run()
