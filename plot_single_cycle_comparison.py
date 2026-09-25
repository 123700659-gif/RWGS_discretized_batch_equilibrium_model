"""Plot one countercurrent and one cocurrent cycle with identical initial states.

Run from the project root with:
    python plot_single_cycle_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from functions.main import simulate_cycle
from functions import build_arrays, build_mu_O_functions


def main():
    output = Path(__file__).resolve().parent / "plots" / "single_cycle_comparison"
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(material="CeZr15", T=1023, n_oxide=10, n_H2=1.0,
                    n_CO2=1.0, x_H2O_0=0.005, x_CO2_0=0.998,
                    gas_mesh=100, oxide_mesh=100)
    _, delta_range = build_arrays.set_delta_max_and_delta_range(settings["material"])
    _, inverse = build_mu_O_functions.set_mu_O_delta_functions(
        settings["material"], settings["T"], delta_range)
    gas_mu = build_mu_O_functions.set_mu_O_CO2_function(
        settings["T"], build_arrays.X_array())
    initial = np.full(settings["oxide_mesh"], float(inverse(gas_mu(settings["x_CO2_0"]))))
    space = np.arange(1, settings["oxide_mesh"] + 1)
    batches = np.arange(1, settings["gas_mesh"] + 1)
    results = {}
    summary = {"settings": settings, "initial_delta": float(initial[0]), "runs": {}}

    for name, left_ox in [("countercurrent", True), ("cocurrent", False)]:
        dr, water, do, co2 = simulate_cycle(**settings, first_cycle=True,
                                            left_flow_red=False, left_flow_ox=left_ox)
        water_out = water[:, -1]
        co2_out = co2[:, 0 if left_ox else -1]
        n_water = float((water_out.mean() - settings["x_H2O_0"]) * settings["n_H2"])
        n_co = float((settings["x_CO2_0"] - co2_out.mean()) * settings["n_CO2"])
        red_transfer = float(settings["n_oxide"] * np.mean(dr[-1] - initial))
        ox_transfer = float(settings["n_oxide"] * np.mean(dr[-1] - do[-1]))
        np.testing.assert_allclose(n_water, red_transfer, atol=1e-12)
        np.testing.assert_allclose(n_co, ox_transfer, atol=1e-12)
        for grid in [dr, water, do, co2]:
            assert np.isfinite(grid).all()
        assert min(water.min(), co2.min()) >= 0 and max(water.max(), co2.max()) <= 1
        results[name] = (dr, water, do, co2)
        summary["runs"][name] = {
            "left_flow_red": False, "left_flow_ox": left_ox,
            "net_H2O_mol": n_water, "net_CO_mol": n_co,
            "reduction_balance_residual_mol": n_water - red_transfer,
            "oxidation_balance_residual_mol": n_co - ox_transfer,
            "final_mean_delta": float(do[-1].mean()),
        }
        np.savez_compressed(output / f"{name}_arrays.npz", initial_delta=initial,
                            delta_red=dr, x_H2O_red=water, delta_ox=do, x_CO2_ox=co2)

    np.testing.assert_array_equal(results["countercurrent"][0], results["cocurrent"][0])
    np.testing.assert_array_equal(results["countercurrent"][1], results["cocurrent"][1])
    delta_top = max(grid.max() for result in results.values() for grid in [result[0], result[2]])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    for name, (dr, water, do, co2) in results.items():
        left_ox = name == "countercurrent"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
        fig.suptitle(f"One {name} cycle | CeZr15 at 1023 K\n"
                     "10 mol oxide; 1 mol per gas feed; 100 solid elements × 100 gas batches",
                     fontsize=14)
        for ax, grid, start, label in [
            (axes[0, 0], dr, initial, "Reduction: H₂ flow →"),
            (axes[0, 1], do, dr[-1], "Oxidation: CO₂ flow ←" if left_ox else "Oxidation: CO₂ flow →"),
        ]:
            ax.plot(space, start, "--", color="#555555", lw=1.8, label="Before phase")
            for batch, color in zip([1, 25, 50, 75, 100], plt.cm.viridis(np.linspace(.1, .9, 5))):
                ax.plot(space, grid[batch - 1], color=color, lw=1.8, label=f"After batch {batch}")
            ax.set(title=label, xlabel="Solid element (fixed left → right position)",
                   ylabel="Oxygen non-stoichiometry δ", xlim=(1,100), ylim=(0, delta_top * 1.08))
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=.18)
        for ax, values, feed, title, ylabel in [
            (axes[1, 0], water[:, -1], settings["x_H2O_0"],
             "Reduction outlet: right end", "H₂O mole fraction"),
            (axes[1, 1], co2[:, 0 if left_ox else -1], settings["x_CO2_0"],
             "Oxidation outlet: left end" if left_ox else "Oxidation outlet: right end", "CO₂ mole fraction"),
        ]:
            ax.plot(batches, values, color="#167d9a", lw=2, label="Outlet")
            ax.axhline(feed, color="#777777", ls="--", lw=1.2, label="Feed")
            ax.set(title=title, xlabel="Gas batch number within phase", ylabel=ylabel,
                   xlim=(1,100), ylim=(-.02,1.02))
            ax.legend(fontsize=9)
            ax.grid(alpha=.18)
        fig.savefig(output / f"{name}.png")
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots and raw arrays saved to {output}")


if __name__ == "__main__":
    main()
