import numpy as np

def batch_equilibrium(mu_O_delta_func, mu_O_gas_func, delta_i, x_i, d_delta, d_X, mu_tolerance=0.1):
    """
    Solve equilibrium condition for a batch reactor in either direction.

    Finds the intersection of solid and gas phase chemical potential
    curves under mass balance constraints.

    The gas is a binary H2O/H2 or CO2/CO mixture. x denotes the mole fraction
    of the oxidiser (H2O or CO2); 1 - x is the mole fraction of the corresponding
    reductant (H2 or CO). This convention applies in either reaction direction.

    Args:
        mu_O_delta_func (callable): Solid-phase oxygen chemical potential as a function of delta.
        mu_O_gas_func (callable): Gas-phase oxygen chemical potential as a function of H2O or CO2 mole fraction.
        delta_i (float): Initial delta value.
        x_i (float): Initial gas-phase oxidiser mole fraction (H2O or CO2).
        d_delta (ndarray): Array of delta value shifts to probe.
        d_X (ndarray): Corresponding probe magnitudes for oxidiser mole fraction changes.
        mu_tolerance (float): Initial equilibrium tolerance in chemical potential units (kJ/mol).

    Returns:
        ndarray: Signed delta shifts; positive for reduction, negative for oxidation.

    Raises:
        ValueError: If no intersection of chemical potential curves is found.
    """
    diff_0 = mu_O_gas_func(x_i) - mu_O_delta_func(delta_i)

    if abs(diff_0) <= mu_tolerance:
        return np.array([0.0])

    if diff_0 < 0:
        # Oxygen leaves the solid: delta and oxidiser fraction x increase.
        diff_mu_O = mu_O_delta_func(delta_i + d_delta) - mu_O_gas_func(x_i + d_X)
        idx = np.argwhere(np.diff(np.sign(diff_mu_O))).flatten()
        if idx.size == 0:
            raise ValueError("Reduction. No intersection of chemical potential curves found.")
        d_d = d_delta[idx] - diff_mu_O[idx] / (
                (diff_mu_O[idx + 1] - diff_mu_O[idx]) / (d_delta[idx + 1] - d_delta[idx])
        )
        return d_d
    else:
        # Oxygen enters the solid: delta and oxidiser fraction x decrease.
        # d_delta contains positive probe magnitudes; the returned d_d is negative.
        diff_mu_O = - mu_O_delta_func(delta_i - d_delta) + mu_O_gas_func(x_i - d_X)
        idx = np.argwhere(np.diff(np.sign(diff_mu_O))).flatten()
        if idx.size == 0:
            raise ValueError("Oxidation. No intersection of chemical potential curves found.")
        d_d = d_delta[idx] - diff_mu_O[idx] / (
        (diff_mu_O[idx + 1] - diff_mu_O[idx]) / (d_delta[idx + 1] - d_delta[idx])
        )
        return -d_d

def oxidation(mu_O_delta_func, mu_O_CO2_func, delta_i, x_CO2_i, d_delta, d_X):
    """
    Solve equilibrium condition for batch reactor oxidation.

    Finds the intersection of chemical potential curves between
    the reduced solid and oxidizing gas stream.

    Args:
        mu_O_delta_func (callable): Solid-phase oxygen chemical potential as a function of delta.
        mu_O_CO2_func (callable): Gas-phase oxygen chemical potential as a function of CO2 mole fraction.
        delta_i (float): Initial delta value.
        x_CO2_i (float): Initial CO2 mole fraction.
        d_delta (ndarray): Array of delta value shifts to probe.
        d_X (ndarray): Corresponding array of gas composition shifts.

    Returns:
        ndarray: Array of delta shifts that satisfy the equilibrium condition.

    Raises:
        ValueError: If no equilibrium point (intersection of chemical potential curves) is found.
    """
    diff_0 = mu_O_CO2_func(x_CO2_i) - mu_O_delta_func(delta_i)

    if diff_0 > 0:
        diff_mu_O = - mu_O_delta_func(delta_i - d_delta) + mu_O_CO2_func(x_CO2_i - d_X)
        idx = np.argwhere(np.diff(np.sign(diff_mu_O))).flatten()
        d_d = d_delta[idx] - diff_mu_O[idx] / (
                (diff_mu_O[idx + 1] - diff_mu_O[idx]) / (d_delta[idx + 1] - d_delta[idx])
        )
        return d_d
    else:
        diff_mu_O = mu_O_delta_func(delta_i + d_delta) - mu_O_CO2_func(x_CO2_i + d_X)
        idx = np.argwhere(np.diff(np.sign(diff_mu_O))).flatten()
        d_d = d_delta[idx] - diff_mu_O[idx] / (
                (diff_mu_O[idx + 1] - diff_mu_O[idx]) / (d_delta[idx + 1] - d_delta[idx])
        )
        return d_d
