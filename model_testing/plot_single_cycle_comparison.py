"""Compare cocurrent, countercurrent, and alternating countercurrent cycles.

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


def main():
    output = Path(__file__).resolve().parent / "plots" / "single_cycle_comparison"
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(material="CeZr15", T=1023, n_oxide=10, n_H2=1.0,
                    n_CO2=1.0, x_H2O_0=0.005, x_CO2_0=0.998,
                    gas_mesh=100, oxide_mesh=100)
    space = np.arange(1, settings["oxide_mesh"] + 1)
    batches = np.arange(1, settings["gas_mesh"] + 1)
    flow_configurations = {
        "countercurrent": ((False, True),),
        "cocurrent": ((False, False),),
        "alternating_countercurrent": ((False, True), (True, False)),
    }
    results = {}
    summary = {"settings": settings, "runs": {}}

    for name, flow_sequence in flow_configurations.items():
        cycle_results = []
        solid_profile = None
        cycle_summaries = []
        for cycle_index, (left_red, left_ox) in enumerate(flow_sequence, start=1):
            cycle_result = simulate_cycle(
                **settings,
                first_cycle=(cycle_index == 1),
                delta_x_0=solid_profile,
                left_flow_red=left_red,
                left_flow_ox=left_ox,
                return_details=True,
            )
            dr, water, do, co2 = cycle_result["grids"]
            start_delta = cycle_result["initial_delta_red"]
            if name == "countercurrent" and cycle_index == 1:
                summary["initial_delta"] = float(start_delta[0])
            water_out = water[:, 0 if left_red else -1]
            co2_out = co2[:, 0 if left_ox else -1]
            n_water = float((water_out.mean() - settings["x_H2O_0"]) * settings["n_H2"])
            n_co = float((settings["x_CO2_0"] - co2_out.mean()) * settings["n_CO2"])
            red_transfer = float(settings["n_oxide"] * np.mean(dr[-1] - start_delta))
            ox_transfer = float(settings["n_oxide"] * np.mean(dr[-1] - do[-1]))
            np.testing.assert_allclose(n_water, red_transfer, atol=1e-12)
            np.testing.assert_allclose(n_co, ox_transfer, atol=1e-12)
            for grid in [dr, water, do, co2]:
                assert np.isfinite(grid).all()
            assert min(water.min(), co2.min()) >= 0 and max(water.max(), co2.max()) <= 1

            cycle_results.append({
                "delta_red": dr,
                "x_H2O_red": water,
                "delta_ox": do,
                "x_CO2_ox": co2,
                "start_delta": start_delta,
                "left_flow_red": left_red,
                "left_flow_ox": left_ox,
            })
            cycle_summaries.append({
                "cycle": cycle_index,
                "left_flow_red": left_red,
                "left_flow_ox": left_ox,
                "net_H2O_mol": n_water,
                "net_CO_mol": n_co,
                "reduction_balance_residual_mol": n_water - red_transfer,
                "oxidation_balance_residual_mol": n_co - ox_transfer,
                "final_mean_delta": float(do[-1].mean()),
            })
            solid_profile = do[-1].copy()
            np.savez_compressed(
                output / f"{name}_cycle_{cycle_index}_arrays.npz",
                initial_delta=start_delta,
                delta_red=dr,
                x_H2O_red=water,
                delta_ox=do,
                x_CO2_ox=co2,
            )

        results[name] = cycle_results
        summary["runs"][name] = {"cycles": cycle_summaries}

    np.testing.assert_array_equal(
        results["countercurrent"][0]["delta_red"],
        results["cocurrent"][0]["delta_red"],
    )
    np.testing.assert_array_equal(
        results["countercurrent"][0]["x_H2O_red"],
        results["cocurrent"][0]["x_H2O_red"],
    )
    delta_top = max(
        grid.max()
        for run in results.values()
        for result in run
        for grid in [result["delta_red"], result["delta_ox"]]
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    batch_indices = np.unique(np.linspace(0, settings["gas_mesh"] - 1, 5, dtype=int))
    for name, run in results.items():
        for cycle_index, result in enumerate(run, start=1):
            dr = result["delta_red"]
            water = result["x_H2O_red"]
            do = result["delta_ox"]
            co2 = result["x_CO2_ox"]
            left_red = result["left_flow_red"]
            left_ox = result["left_flow_ox"]
            figure_name = (
                f"{name}_cycle_{cycle_index}" if len(run) > 1 else name
            )
            fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
            # fig.suptitle(f"{name.replace('_', ' ').title()} cycle {cycle_index} | CeZr15 at 1023 K\n"
            #             "10 mol oxide; 1 mol per gas feed; 100 solid elements × 100 gas batches",
            #             fontsize=14)
            for ax, grid, start, label in [
                (axes[0, 0], dr, result["start_delta"],
                 f"Reduction: H₂ flow {'←' if left_red else '→'}"),
                (axes[0, 1], do, dr[-1],
                 f"Oxidation: CO₂ flow {'←' if left_ox else '→'}"),
            ]:
                ax.plot(space, start, "--", color="#555555", lw=1.8, label="Before phase")
                for batch, color in zip(batch_indices, plt.cm.viridis(np.linspace(.1, .9, len(batch_indices)))):
                    ax.plot(space, grid[batch], color=color, lw=1.8,
                            label=f"After batch {batch + 1}")
                ax.set(title=label, xlabel="Solid element (fixed left → right position)",
                       ylabel="Oxygen non-stoichiometry δ", xlim=(1, settings["oxide_mesh"]),
                       ylim=(0, delta_top * 1.08))
                ax.legend(fontsize=8, ncol=2)
                ax.grid(alpha=.18)
            for ax, values, feed, title, ylabel in [
                (axes[1, 0], water[:, 0 if left_red else -1], settings["x_H2O_0"],
                 f"Reduction outlet: {'left' if left_red else 'right'} end", "H₂O mole fraction"),
                (axes[1, 1], co2[:, 0 if left_ox else -1], settings["x_CO2_0"],
                 f"Oxidation outlet: {'left' if left_ox else 'right'} end", "CO₂ mole fraction"),
            ]:
                ax.plot(batches, values, color="#167d9a", lw=2, label="Outlet")
                ax.axhline(feed, color="#777777", ls="--", lw=1.2, label="Feed")
                ax.set(title=title, xlabel="Gas batch number within phase", ylabel=ylabel,
                       xlim=(1, settings["gas_mesh"]), ylim=(-.02, 1.02))
                ax.legend(fontsize=9)
                ax.grid(alpha=.18)
            fig.savefig(output / f"{figure_name}.png")
            fig.savefig(output / f"{figure_name}.svg")
            plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Plots and raw arrays saved to {output}")


if __name__ == "__main__":
    main()
