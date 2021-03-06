include("../src/feautrier.jl")
include("../src/populations.jl")


function run()
    println("\n", "="^91, "\n", " "^34,
            "SOLAR ATMOSPHERE MCRT",
            "\n", "="^91, "\n")

    # =============================================================================
    # LOAD ATMOSPHERE DATA
    # =============================================================================
    print("--Loading atmosphere data..................")
    atmosphere_parameters = collect_atmosphere_data()
    atmosphere = Atmosphere(atmosphere_parameters...)
    atmosphere_size = size(atmosphere.temperature)
    println("Atmosphere loaded with dimensions ", atmosphere_size, ".")

    if background_mode()
        # =============================================================================
        # READ CONFIG FILE
        # =============================================================================
        output_path = get_output_path()
        nμ, nϕ = get_angles()

        # =============================================================================
        # LOAD WAVELENGTH
        # =============================================================================
        print("--Loading wavelength.......................")
        λ = get_background_λ()
        nλ = length(λ)
        println("Wavelength λ = ", λ, " loaded.")

        # =============================================================================
        # LOAD RADIATION DATA
        # =============================================================================
        print("--Loading radiation data...................")
        radiation_parameters = collect_radiation_data(atmosphere, λ)
        radiation = RadiationBackground(radiation_parameters...)
        println("Radiation loaded.")

        # =============================================================================
        # CREATE OUTPUT FILE
        # =============================================================================
        print("--Initialise output file...................")
        create_output_file(output_path, nλ, atmosphere_size)
        write_to_file([λ], output_path)
        println(@sprintf("%.1f GBs of data initialised.", how_much_data(nλ, atmosphere_size)))

        # =============================================================================
        # SIMULATION
        # =============================================================================
        feautrier(atmosphere, radiation, λ, nμ, nϕ, output_path)

        # =============================================================================
        # END OF TEST MODE
        # =============================================================================
    else
        # =============================================================================
        # READ CONFIG FILE
        # =============================================================================
        output_path = get_output_path()
        max_iterations = get_max_iterations()
        population_distribution = get_population_distribution()
        write_rates = get_write_rates()
        nμ, nϕ = get_angles()

        # =============================================================================
        # LOAD ATOM
        # =============================================================================
        print("--Loading atom.............................")
        atom_parameters = collect_atom_data(atmosphere)
        atom = Atom(atom_parameters...)
        nλ = atom.nλ_bb + 2*atom.nλ_bf
        println("Atom loaded with ", nλ , " wavelengths.")

        # =============================================================================
        # LOAD INITIAL POPULATIONS
        # =============================================================================
        print("--Loading initial populations..............")
        populations = collect_initial_populations(atom, atmosphere.temperature,
                                                        atmosphere.electron_density)
        println("Initial ", population_distribution, "-populations loaded.")

        # =============================================================================
        # CALCULATE INITIAL TRANSITION RATES
        # =============================================================================
        print("--Loading initial transition rates.........")
        Bλ = blackbody_lambda(atom.λ, atmosphere.temperature)
        rate_parameters = calculate_transition_rates(atom, atmosphere.temperature,
                                                           atmosphere.electron_density, Bλ)
        rates = TransitionRates(rate_parameters...)
        println("Initial transition rates loaded.")

        # =============================================================================
        # CREATE OUTPUT FILE
        # =============================================================================
        print("--Initialise output file...................")
        create_output_file(output_path, max_iterations, nλ, atmosphere_size, write_rates)
        write_to_file(atom, output_path)
        write_to_file(populations, 0, output_path)
        if write_rates; write_to_file(rates, 0, output_path); end
        println(@sprintf("%.1f GBs of data initialised.", how_much_data(nλ, atmosphere_size, max_iterations, write_rates)))

        # =============================================================================
        # RUN MCRT UNTIL POPULATIONS CONVERGE
        # =============================================================================
        converged_populations = false

        for n=1:max_iterations
            println("\n  ITERATION ", n, "\n", "="^91)
            # =============================================================================
            # LOAD RADIATION DATA WITH CURRENT POPULATIONS
            # =============================================================================
            print("--Loading radiation data...................")
            radiation_parameters = collect_radiation_data(atmosphere, atom, rates, populations)
            radiation = Radiation(radiation_parameters...)
            println("Radiation loaded.")

            # =============================================================================
            # SIMULATION
            # =============================================================================
            feautrier(atmosphere, radiation, atom, nμ, nϕ, n, output_path)

            # =============================================================================
            # CALCULATE NEW TRANSITION RATES
            # =============================================================================
            print("\n--Update transition rates..................")
            Jλ = get_Jλ(output_path, n)
            rate_parameters = calculate_transition_rates(atom, atmosphere.temperature,
                                                               atmosphere.electron_density, Jλ)
            rates = TransitionRates(rate_parameters...)
            if write_rates; write_to_file(rates, n, output_path); end
            println("Transition rates updated.")

            # =============================================================================
            # CALCULATE NEW POPULATIONS
            # =============================================================================
            print("--Update populations.......................")
            new_populations = get_revised_populations(rates, atom.density)
            write_to_file(new_populations, n, output_path)
            println("Populations updated.")

            # =============================================================================
            # CHECK POPULATION CONVERGENCE
            # =============================================================================
            converged, error = check_population_convergence(populations, new_populations)
            populations = copy(new_populations)

            if converged
                println(@sprintf("--Convergence at iteration n = %d. Error = %.1e.\n", n, error))
                cut_output_file(output_path, n, write_rates)
                break
            else
                println(@sprintf("\n--No convergence. Error = %.1e.\n", error))
            end

            # =============================================================================
            # END OF ITERATION
            # =============================================================================
        end

        # =============================================================================
        # END OF ATOM MODE
        # =============================================================================
    end

end

run()
