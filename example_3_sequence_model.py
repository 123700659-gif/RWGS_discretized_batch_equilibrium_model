"""Sweep oxide loading using a fixed flow sequence and H2/CO2 feed ratio."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


from functions.main import cycle_until_balanced


# === Process conditions
material = "CeZr20"
T = 1023  # Temperature [K]
n_CO2 = 1.0  # Moles of CO2 per oxidation cycle
H2_CO2_ratio = 1.5
n_H2 = H2_CO2_ratio * n_CO2
x_CO2_0 = 0.998
x_H2O_0 = 0.005
max_cycles = 60
oxide_moles = np.linspace(2, 100, 20)

# Run the full number of requested cycles (only complete sequence repetitions
# run), even if the sequence balances and the solid profile converges sooner.
flow_configurations = [
    ("Alternating countercurrent", [(False, True), (True, False)]),
    ("Countercurrent", [(False, True)]),
    ("Cocurrent", [(False, False)]),
]

def main():
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for configuration_name, flow_sequence in flow_configurations:
        co2_conversions = []
        for n_oxide in oxide_moles:
            result = cycle_until_balanced(
                max_cycles=max_cycles,
                material=material,
                T=T,
                n_CO2=n_CO2,
                n_H2=n_H2,
                n_oxide=float(n_oxide),
                x_CO2_0=x_CO2_0,
                x_H2O_0=x_H2O_0,
                flow_sequence=flow_sequence,
                return_details=True,
                stop_on_convergence=False,
            )

            # Use the final complete flow sequence, even if the model did not
            # converge before reaching max_cycles.
            sequence_CO = sum(cycle["n_CO"] for cycle in result["sequence_results"])
            conversion = sequence_CO / (n_CO2 * len(flow_sequence))
            co2_conversions.append(conversion)

            status = "balanced" if result["converged"] else "max cycles"
            print(
                f"{configuration_name}, n_oxide={n_oxide:.2f}, "
                f"cycles={result['cycles']}, "
                f"status={status}, X_CO2={conversion:.4f}"
            )

        ax.plot(
            oxide_moles,
            co2_conversions,
            marker="o",
            linewidth=1.4,
            label=configuration_name,
        )

    ax.set_xlabel("Moles of oxide")
    ax.set_ylabel(r"CO$_2$ conversion $X_{\mathrm{CO_2}}$")
    ax.set_title(f"{material}, {T} K, H$_2$/CO$_2$ = {H2_CO2_ratio:g}")
    ax.set_xlim(oxide_moles[0], oxide_moles[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
