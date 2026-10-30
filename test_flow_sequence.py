"""Run with python -m unittest test_flow_sequence."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from functions.batch_equilibrium_solver import EquilibriumDomainError
from functions.main import cycle_until_balanced, simulate_cycle


class FlowSequenceTests(unittest.TestCase):
    def test_first_cycle_saturates_out_of_range_feed_and_reports_domain(self):
        def unchanged_phase(_mu_O_delta_func, _mu_O_gas_func, x_0,
                            delta_t_x, x_gas_t_x, *_args, **_kwargs):
            delta_t_x[:] = delta_t_x[0]
            x_gas_t_x[:] = x_0
            _ = (_mu_O_delta_func, _mu_O_gas_func, _args, _kwargs)
            return delta_t_x, x_gas_t_x

        with patch('functions.main.sequential_batch_equilibrium',
                   side_effect=unchanged_phase), self.assertWarns(RuntimeWarning):
            result = simulate_cycle(
                T=500, oxide_mesh=2, gas_mesh=2, return_details=True,
            )

        diagnostic = result['initial_equilibrium_diagnostic']
        self.assertTrue(diagnostic['outside_solid_domain'])
        self.assertEqual(diagnostic['direction'], 'above')
        np.testing.assert_array_equal(
            result['initial_delta_red'],
            np.full(2, diagnostic['initial_delta']),
        )
        self.assertAlmostEqual(diagnostic['initial_delta'], 1e-6)

    def test_first_cycle_uses_reduced_endpoint_when_feed_is_below_range(self):
        def unchanged_phase(_mu_O_delta_func, _mu_O_gas_func, x_0,
                            delta_t_x, x_gas_t_x, *_args, **_kwargs):
            delta_t_x[:] = delta_t_x[0]
            x_gas_t_x[:] = x_0
            _ = (_mu_O_delta_func, _mu_O_gas_func, _args, _kwargs)
            return delta_t_x, x_gas_t_x

        with patch('functions.main.sequential_batch_equilibrium',
                   side_effect=unchanged_phase), self.assertWarns(RuntimeWarning):
            result = simulate_cycle(
                T=1500, x_CO2_0=1e-6, oxide_mesh=2, gas_mesh=2,
                return_details=True,
            )

        diagnostic = result['initial_equilibrium_diagnostic']
        self.assertTrue(diagnostic['outside_solid_domain'])
        self.assertEqual(diagnostic['direction'], 'below')
        self.assertAlmostEqual(diagnostic['initial_delta'], 0.335)

    def test_batch_failure_keeps_initial_out_of_domain_diagnostic(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(EquilibriumDomainError) as raised:
                simulate_cycle(T=500, oxide_mesh=4, gas_mesh=5)

        diagnostic = raised.exception.initial_equilibrium_diagnostic
        self.assertTrue(diagnostic['outside_solid_domain'])
        self.assertEqual(diagnostic['direction'], 'above')
        self.assertIn('lies above the modeled solid domain', str(raised.exception))

    def test_sequence_balance_outlets_and_profile_handoff(self):
        calls = []

        def fake_cycle(**kwargs):
            calls.append(kwargs)
            left_red, left_ox = kwargs['left_flow_red'], kwargs['left_flow_ox']
            # Each pair is unbalanced; only the full sequence balances.
            water, co = (0.1, 0.2) if not left_red else (0.2, 0.1)
            red = np.full((2, 3), 0.3)
            ox = np.tile([0.1, 0.2, 0.3], (2, 1))
            h2o = np.full((2, 3), -99.0)
            co2 = np.full((2, 3), -99.0)
            h2o[:, 0 if left_red else -1] = kwargs['x_H2O_0'] + water
            co2[:, 0 if left_ox else -1] = kwargs['x_CO2_0'] - co
            return red, h2o, ox, co2

        sequence = [(False, True), (True, False)]
        with patch('functions.main.simulate_cycle', side_effect=fake_cycle):
            result = cycle_until_balanced(
                flow_sequence=sequence, max_cycles=8, n_H2=1, n_CO2=1,
                return_details=True,
            )
        self.assertEqual(result['cycles'], 4)
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['n_H2O'], 0.3)
        self.assertAlmostEqual(result['n_CO'], 0.3)
        self.assertEqual([(c['left_flow_red'], c['left_flow_ox']) for c in calls], sequence * 2)
        self.assertTrue(calls[0]['first_cycle'])
        for call in calls[1:]:
            self.assertFalse(call['first_cycle'])
            np.testing.assert_array_equal(call['delta_x_0'], [0.1, 0.2, 0.3])
        self.assertEqual(len(result['sequence_results']), 2)

    def test_balanced_gas_is_insufficient_when_profile_changes(self):
        count = 0

        def changing_cycle(**kwargs):
            nonlocal count
            count += 1
            solid = np.full((1, 2), count * 0.01)
            return (solid.copy(), np.full((1, 2), kwargs['x_H2O_0']),
                    solid, np.full((1, 2), kwargs['x_CO2_0']))

        with patch('functions.main.simulate_cycle', side_effect=changing_cycle):
            with self.assertWarns(RuntimeWarning):
                result = cycle_until_balanced(
                    flow_sequence=[(False, True), (True, False)],
                    max_cycles=5, return_details=True,
                )
        self.assertEqual(count, 4)  # Never stop halfway through a sequence.
        self.assertFalse(result['converged'])
        self.assertEqual(result['oxygen_balance_error'], 0.0)
        self.assertAlmostEqual(result['profile_error'], 0.02)

    def test_can_run_all_requested_cycles_after_convergence(self):
        def balanced_cycle(**kwargs):
            solid = np.full((1, 2), 0.01)
            gas = np.full((1, 2), kwargs['x_H2O_0'])
            co2 = np.full((1, 2), kwargs['x_CO2_0'])
            return solid.copy(), gas, solid.copy(), co2

        with patch('functions.main.simulate_cycle', side_effect=balanced_cycle):
            result = cycle_until_balanced(
                flow_sequence=[(False, True), (True, False)],
                max_cycles=6, return_details=True, stop_on_convergence=False,
            )
        self.assertEqual(result['cycles'], 6)
        self.assertTrue(result['converged'])
        self.assertEqual(len(result['sequence_results']), 2)

    def test_real_cycles_preserve_oxygen_and_orientation(self):
        for red_left, ox_left in [(False, False), (False, True), (True, False), (True, True)]:
            initial = np.linspace(0.01, 0.02, 4)
            dr, water, do, co2 = simulate_cycle(
                first_cycle=False, delta_x_0=initial, oxide_mesh=4, gas_mesh=5,
                left_flow_red=red_left, left_flow_ox=ox_left,
            )
            n_water = (water[:, 0 if red_left else -1].mean() - 0.005) * 1.01
            n_co = (0.998 - co2[:, 0 if ox_left else -1].mean())
            self.assertAlmostEqual(n_water, 20 * np.mean(dr[-1] - initial), places=10)
            self.assertAlmostEqual(n_co, 20 * np.mean(dr[-1] - do[-1]), places=10)
            np.testing.assert_array_equal(initial, np.linspace(0.01, 0.02, 4))

    def test_legacy_return_and_sequence_validation(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = cycle_until_balanced(max_cycles=2, oxide_mesh=3, gas_mesh=4)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0].shape, (4, 3))
        for sequence in [[], [(False,)], [('right', 'left')]]:
            with self.assertRaises(ValueError):
                cycle_until_balanced(flow_sequence=sequence)
        with self.assertRaises(ValueError):
            cycle_until_balanced(max_cycles=1, flow_sequence=[(False, True)] * 2)


if __name__ == '__main__':
    unittest.main()
