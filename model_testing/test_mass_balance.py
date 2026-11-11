"""Run with python -m unittest test_mass_balance."""
import unittest
import numpy as np

from functions.main import simulate_cycle
from functions.mass_balance import calculate_mass_balance

class BalanceTests(unittest.TestCase):
    def test_directions_and_initial_state(self):
        n_oxide, n_H2, n_CO2 = 10, 1.5, .8
        x_H2O_0, x_CO2_0 = 0.005, 0.998
        for red in (False, True):
            for ox in (False, True):
                for first in (False, True):
                    initial = np.linspace(.01, .02, 7)
                    result = simulate_cycle(material='CeZr15', T=1023, n_oxide=n_oxide,
                        n_H2=n_H2, n_CO2=n_CO2, x_H2O_0=x_H2O_0, x_CO2_0=x_CO2_0,
                        gas_mesh=9, oxide_mesh=7,
                        first_cycle=first, delta_x_0=initial,
                        left_flow_red=red, left_flow_ox=ox, return_details=True)
                    np.testing.assert_array_equal(initial, np.linspace(.01, .02, 7))
                    delta_red, _, delta_ox, _ = result['grids']
                    X_CO2, X_H2, _, O_bal_CO2_oxide, O_bal_H2_oxide = calculate_mass_balance(
                        *result['grids'], initial_delta_red=result['initial_delta_red'],
                        n_oxide=n_oxide, n_H2=n_H2, n_CO2=n_CO2,
                        x_CO2_0=x_CO2_0, x_H2O_0=x_H2O_0,
                        left_flow_red=red, left_flow_ox=ox)

                    # Gas-side oxygen equals oxide-side oxygen in each step
                    nO_red = n_oxide * np.mean(delta_red[-1] - result['initial_delta_red'])
                    nO_ox = n_oxide * np.mean(delta_red[-1] - delta_ox[-1])
                    self.assertLess(abs(X_H2 * n_H2 - nO_red), 1e-12)
                    self.assertLess(abs(X_CO2 * n_CO2 - nO_ox), 1e-12)
                    self.assertAlmostEqual(O_bal_H2_oxide, 1, places=10)
                    self.assertAlmostEqual(O_bal_CO2_oxide, 1, places=10)


if __name__ == '__main__':
    unittest.main()
