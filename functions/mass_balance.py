import numpy as np

def calculate_mass_balance(delta_t_x_red, x_H2O_t_x_red, delta_t_x_ox, x_CO2_t_x_ox,
                           n_CO2=1.0, n_H2=1.0, n_oxide=20.0, x_CO2_0=0.998, x_H2O_0=0.005,
                           left_flow_red=False, left_flow_ox=True):
    """
    Compute the mass balance for a full redox cycle.

    The calculation compares gas conversion and solid oxygen exchange
    during reduction and oxidation steps.

    Args:
        delta_t_x_red (ndarray): Delta grid after reduction.
        x_H2O_t_x_red (ndarray): H2O mole fraction grid after reduction.
        delta_t_x_ox (ndarray): Delta grid after oxidation.
        x_CO2_t_x_ox (ndarray): CO2 mole fraction grid after oxidation.
        n_CO2 (float): Moles of CO2 used in oxidation.
        n_H2 (float): Moles of hydrogen used in reduction.
        n_oxide (float): Moles of oxide in the reactor.
        x_CO2_0 (float): Initial mole fraction of CO2.
        x_H2O_0 (float): Initial mole fraction of H2O.

    Returns:
        tuple:
            X_CO2 (float): Net CO2 conversion.
            X_H2 (float): Apparent H2 conversion inferred from H2O production.
            O_balance_CO_H2O (float): Oxygen mass balance between CO2 and H2 streams.
            O_balance_CO2_oxide (float): Oxygen mass balance between CO2 stream and oxide.
    """
    # Infer H2 consumption from H2O production using the 1:1 H2:H2O stoichiometry.
    x_H2O_outlet = x_H2O_t_x_red.T[-1].mean()
    X_H2 = x_H2O_outlet - x_H2O_0
    n_H2O = X_H2 * n_H2

    # Change in delta across the reduction-to-oxidation endpoints.
    d_delta_material = delta_t_x_red[-1] - delta_t_x_ox[-1]
    nO_material = (d_delta_material.sum() * n_oxide / len(d_delta_material))    
    # Total moles of CO produced
    if left_flow_ox:
        x_CO2_in = x_CO2_t_x_ox[:, -1]
        x_CO2_out = x_CO2_t_x_ox[:, 0]
    elif not left_flow_ox:
        x_CO2_in = x_CO2_t_x_ox[:, 0]
        x_CO2_out = x_CO2_t_x_ox[:, -1]

    X_CO2 = np.mean(x_CO2_in - x_CO2_out)
    n_CO = X_CO2 * n_CO2

    # Oxygen atom balance comparisons
    O_balance_CO_H2O = n_CO / n_H2O if n_H2O != 0 else np.nan  # gas-to-gas oxygen balance
    O_balance_CO2_oxide = n_CO / nO_material if nO_material != 0 else np.nan  # gas-to-solid oxygen balance

    return X_CO2, X_H2, O_balance_CO_H2O, O_balance_CO2_oxide
