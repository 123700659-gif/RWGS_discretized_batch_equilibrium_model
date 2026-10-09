import numpy as np


def calculate_mass_balance(delta_t_x_red, x_H2O_t_x_red,
                           delta_t_x_ox, x_CO2_t_x_ox,
                           initial_delta_red,
                           n_CO2=1.0, n_H2=1.0, n_oxide=20.0,
                           x_CO2_0=0.998, x_H2O_0=0.005,
                           left_flow_red=False, left_flow_ox=True):
    """
    Compute the oxygen mass balances for a full redox cycle.

    The calculation compares gas conversion with oxygen exchanged
    by the solid during reduction and oxidation.

    Args:
        delta_t_x_red (ndarray): Delta grid after reduction.
        x_H2O_t_x_red (ndarray): H2O mole fraction grid during reduction.
        delta_t_x_ox (ndarray): Delta grid after oxidation.
        x_CO2_t_x_ox (ndarray): CO2 mole fraction grid during oxidation.
        initial_delta_red (ndarray): Delta profile before reduction.
        n_CO2 (float): Moles of CO2/CO gas used in oxidation.
        n_H2 (float): Moles of H2/H2O gas used in reduction.
        n_oxide (float): Moles of oxide in the reactor.
        x_CO2_0 (float): Inlet mole fraction of CO2.
        x_H2O_0 (float): Inlet mole fraction of H2O.
        left_flow_red (bool): Reduction flow direction.
        left_flow_ox (bool): Oxidation flow direction.

    Returns:
        X_CO2 (float): Net CO2 conversion.
        X_H2 (float): Net H2 conversion.
        O_balance_CO_H2O (float): Oxygen balance between CO2 and H2 streams.
        O_balance_CO2_oxide (float): Oxygen balance during oxidation.
        O_balance_H2_oxide (float): Oxygen balance during reduction.
    """

    # Physical gas outlets
    outlet_red = 0 if left_flow_red else -1
    outlet_ox = 0 if left_flow_ox else -1

    # Gas conversions
    X_CO2 = x_CO2_0 - x_CO2_t_x_ox[:, outlet_ox].mean()
    X_H2 = x_H2O_t_x_red[:, outlet_red].mean() - x_H2O_0

    # Total moles of H2O and CO produced
    n_H2O = X_H2 * n_H2
    n_CO = X_CO2 * n_CO2

    # Oxygen removed from the oxide during reduction
    d_delta_red = delta_t_x_red[-1] - initial_delta_red
    nO_oxide_red = d_delta_red.mean() * n_oxide

    # Oxygen added to the oxide during oxidation
    d_delta_ox = delta_t_x_red[-1] - delta_t_x_ox[-1]
    nO_oxide_ox = d_delta_ox.mean() * n_oxide

    # Oxygen atom balance comparisons
    O_balance_CO_H2O = n_CO / n_H2O
    O_balance_CO2_oxide = n_CO / nO_oxide_ox
    O_balance_H2_oxide = n_H2O / nO_oxide_red

    return X_CO2, X_H2, O_balance_CO_H2O, O_balance_CO2_oxide, O_balance_H2_oxide
