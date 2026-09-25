import numpy as np
from functions import build_arrays
from functions import build_mu_O_functions
from functions.discretized_solver import sequential_batch_equilibrium

def simulate_cycle(material="CeO2", first_cycle=True, delta_x_0=None,
                   T=1073, n_CO2=1.0, n_H2=1.01, n_oxide=20,
                   x_H2O_0=0.005, x_CO2_0=0.998,
                   oxide_mesh=100, gas_mesh=100,
                   left_flow_red=False, left_flow_ox=True):
    """
    Simulate a single chemical looping cycle (reduction + oxidation) in a 1D reactor.

    Gas composition x always denotes the oxidiser mole fraction: x_H2O for
    the H2O/H2 mixture and x_CO2 for the CO2/CO mixture. The corresponding
    H2 and CO mole fractions are 1 - x_H2O and 1 - x_CO2. These are binary
    gas mixtures, and this convention also applies during local reverse reactions.

    Args:
        material (str): Material name (options, "CeO2", "LSF", "CeO2_simple", "CeZr05", "CeZr15").
        first_cycle (bool): True if this is the first cycle (used to set starting codition).
        delta_x_0 (ndarray): Initial delta profile if not first cycle.
        T (float): Operating temperature in Kelvin.
        n_CO2 (float): Moles of CO2 used in oxidation.
        n_H2 (float): Moles of H2 used in reduction.
        n_oxide (float): Moles of solid oxide in the reactor.
        x_H2O_0 (float): Initial mole fraction of H2O in feed gas.
        x_CO2_0 (float): Initial mole fraction of CO2 in feed gas.
        oxide_mesh (int): Number of discretization points in solid.
        gas_mesh (int): Number of discretization points in gas (time-like).
        left_flow_red (bool): Reduction flows left if True, right otherwise.
        left_flow_ox (bool): Oxidation flows left if True, right otherwise.

    All returned grids use the same physical spatial orientation. No additional
    flips are needed when passing the final solid profile to the next cycle.

    Returns:
        tuple:
            - delta_t_x_red (ndarray): Delta solution matrix after reduction.
            - x_H2O_t_x_red (ndarray): H2O solution matrix after reduction.
            - delta_t_x_ox (ndarray): Delta solution matrix after oxidation.
            - x_CO2_t_x_ox (ndarray): CO2 solution matrix after oxidation.
    """
    # Mass balance scaling factors
    mbf_red = (n_oxide / n_H2) * (gas_mesh / oxide_mesh)
    mbf_ox = (n_oxide / n_CO2) * (gas_mesh / oxide_mesh)

    # Set delta_max and delta range
    delta_max, delta_range = build_arrays.set_delta_max_and_delta_range(material)

    # Generate mu_O(delta) function for the oxide and its inverse
    mu_O_delta_func, mu_O_delta_func_inv = build_mu_O_functions.set_mu_O_delta_functions(material, T, delta_range)

    # Generate X (gas composition) range
    X_range = build_arrays.X_array()

    # Generate mu_O(X) functions for gas streams
    mu_O_CO2_func = build_mu_O_functions.set_mu_O_CO2_function(T, X_range)
    mu_O_H2O_func = build_mu_O_functions.set_mu_O_H2O_function(T, X_range)

    # Use the oxidizing feed equilibrium only to initialize the first cycle.
    max_oxidising_mu_O = mu_O_CO2_func(x_CO2_0)
    delta_min = mu_O_delta_func_inv(max_oxidising_mu_O)
    # Retain the material range for probing either reaction direction. This
    # sets probe magnitudes; it does not enforce final-state physical bounds.

    # Initialize simulation arrays
    delta_t_x_red = np.zeros((oxide_mesh, gas_mesh)).T
    x_H2O_t_x_red = np.zeros((oxide_mesh, gas_mesh)).T
    delta_t_x_ox = np.zeros((oxide_mesh, gas_mesh)).T
    x_CO2_t_x_ox = np.zeros((oxide_mesh, gas_mesh)).T

    # Initial condition for reduction
    if first_cycle:
        delta_t_x_red[0] = delta_min
    else:
        initial_profile = np.asarray(delta_x_0, dtype=float)
        if initial_profile.shape != (oxide_mesh,):
            raise ValueError("delta_x_0 must contain one value per oxide element.")
        delta_t_x_red[0] = initial_profile

    # Create differential change in oxygen mass balance arrays for batch equilibrium solver
    d_delta_red, d_X_red = build_arrays.mass_balance_arrays(
        delta_max=delta_max, n_gas=n_H2,
        n_oxide=n_oxide, gas_mesh=gas_mesh, oxide_mesh=oxide_mesh
    )
    d_delta_ox, d_X_ox = build_arrays.mass_balance_arrays(
        delta_max=delta_max, n_gas=n_CO2,
        n_oxide=n_oxide, gas_mesh=gas_mesh, oxide_mesh=oxide_mesh
    )

    # Run reduction simulation
    delta_t_x_red, x_H2O_t_x_red = sequential_batch_equilibrium(
        mu_O_delta_func, mu_O_H2O_func, x_H2O_0,
        delta_t_x_red, x_H2O_t_x_red, d_delta_red, d_X_red,
        mbf_red, left_flow=left_flow_red
    )

    # Use reduction output as initial condition for oxidation
    delta_t_x_ox[0] = delta_t_x_red[-1]

    # Run oxidation simulation
    delta_t_x_ox, x_CO2_t_x_ox = sequential_batch_equilibrium(
        mu_O_delta_func, mu_O_CO2_func, x_CO2_0,
        delta_t_x_ox, x_CO2_t_x_ox, d_delta_ox, d_X_ox,
        mbf_ox, left_flow=left_flow_ox
    )

    return delta_t_x_red, x_H2O_t_x_red, delta_t_x_ox, x_CO2_t_x_ox


def cycle_until_balanced(max_cycles=20, O_balance_tolerance=0.001, material="CeO2",
                         T=1073, n_CO2=1.0, n_H2=1.01, n_oxide=20,
                         x_H2O_0=0.005, x_CO2_0=0.998,
                         oxide_mesh=100, gas_mesh=100,
                         flow_sequence=None, delta_tolerance=1e-4,
                         return_details=False):
    """Repeat a flow sequence until its oxygen balance and solid profile repeat.

    x_H2O_0 and x_CO2_0 are inlet oxidiser mole fractions. The remaining inlet
    fraction is H2 or CO, respectively. Returned gas grids follow the same
    convention: the reductant fraction is 1 minus the stored oxidiser fraction.

    flow_sequence contains (left_flow_red, left_flow_ox) boolean pairs.
    The default [(False, True)] is countercurrent. [(False, False)] is
    cocurrent. [(False, True), (True, False)] alternates right reduction,
    left oxidation, left reduction, right oxidation.

    max_cycles limits reduction-oxidation pairs, not sequence repetitions.
    Only complete sequences run; unused capacity smaller than one sequence
    is ignored. At least two repetitions are needed to establish convergence.
    n_H2 and n_CO2 are feed amounts per phase, not per complete sequence.

    Convergence requires both:
      * abs(n_CO - n_H2O) / max(abs(n_CO), abs(n_H2O)) <= O_balance_tolerance,
        using net product totals over the complete sequence (zero if both vanish).
      * Maximum absolute change in final delta profile between successive
        complete sequences <= delta_tolerance.

    By default returns the last pair's four grids and total number of pairs,
    preserving the existing five-value return format. With return_details=True,
    returns a dictionary containing every pair of the final complete sequence,
    its flow directions and net products, convergence status, and diagnostics
    for every repetition. Earlier repetitions' full grids are not retained.
    Reaching max_cycles without convergence emits a warning for either format.
    """
    import warnings

    if flow_sequence is None:
        flow_sequence = [(False, True)]
    sequence = list(flow_sequence)
    if not sequence:
        raise ValueError("flow_sequence must contain at least one direction pair.")
    for pair in sequence:
        if len(pair) != 2 or not all(isinstance(flag, (bool, np.bool_)) for flag in pair):
            raise ValueError("Each flow pair must contain two booleans: reduction, oxidation.")
    if not isinstance(max_cycles, (int, np.integer)) or max_cycles < len(sequence):
        raise ValueError("max_cycles must allow at least one complete flow sequence.")
    if not np.isfinite(O_balance_tolerance) or O_balance_tolerance < 0:
        raise ValueError("O_balance_tolerance must be finite and nonnegative.")
    if not np.isfinite(delta_tolerance) or delta_tolerance < 0:
        raise ValueError("delta_tolerance must be finite and nonnegative.")

    initial_profile = None
    previous_sequence_end = None
    cycles = 0
    converged = False
    history = []

    for repetition in range(1, max_cycles // len(sequence) + 1):
        sequence_results = []
        total_H2O = 0.0
        total_CO = 0.0

        for left_flow_red, left_flow_ox in sequence:
            grids = simulate_cycle(
                material=material, first_cycle=(cycles == 0),
                delta_x_0=initial_profile,
                T=T, n_CO2=n_CO2, n_H2=n_H2, n_oxide=n_oxide,
                x_H2O_0=x_H2O_0, x_CO2_0=x_CO2_0,
                oxide_mesh=oxide_mesh, gas_mesh=gas_mesh,
                left_flow_red=left_flow_red, left_flow_ox=left_flow_ox,
            )
            delta_red, x_H2O_red, delta_ox, x_CO2_ox = grids
            initial_profile = delta_ox[-1].copy()
            outlet_red = 0 if left_flow_red else -1
            outlet_ox = 0 if left_flow_ox else -1
            n_H2O = float((x_H2O_red[:, outlet_red].mean() - x_H2O_0) * n_H2)
            n_CO = float((x_CO2_0 - x_CO2_ox[:, outlet_ox].mean()) * n_CO2)
            total_H2O += n_H2O
            total_CO += n_CO
            cycles += 1
            sequence_results.append({
                "cycle": cycles,
                "left_flow_red": bool(left_flow_red),
                "left_flow_ox": bool(left_flow_ox),
                "delta_red": delta_red, "x_H2O_red": x_H2O_red,
                "delta_ox": delta_ox, "x_CO2_ox": x_CO2_ox,
                "n_H2O": n_H2O, "n_CO": n_CO,
            })

        scale = max(abs(total_H2O), abs(total_CO))
        oxygen_balance_error = abs(total_CO - total_H2O) / scale if scale else 0.0
        profile_error = (
            float(np.max(np.abs(initial_profile - previous_sequence_end)))
            if previous_sequence_end is not None else float("inf")
        )
        converged = bool(
            oxygen_balance_error <= O_balance_tolerance
            and profile_error <= delta_tolerance
        )
        history.append({
            "repetition": repetition, "cycles": cycles,
            "n_H2O": total_H2O, "n_CO": total_CO,
            "oxygen_balance_error": oxygen_balance_error,
            "profile_error": profile_error,
            "converged": converged,
        })
        if converged:
            break
        previous_sequence_end = initial_profile.copy()

    if not converged:
        warnings.warn(
            f"No sequence convergence after {cycles} cycles ({repetition} complete sequences).",
            RuntimeWarning, stacklevel=2,
        )
    if return_details:
        return {
            "flow_sequence": [tuple(map(bool, pair)) for pair in sequence],
            "sequence_results": sequence_results,
            "cycles": cycles, "sequence_repetitions": repetition,
            "converged": converged,
            "oxygen_balance_error": oxygen_balance_error,
            "profile_error": profile_error,
            "n_H2O": total_H2O, "n_CO": total_CO,
            "history": history,
        }
    return (*grids, cycles)
