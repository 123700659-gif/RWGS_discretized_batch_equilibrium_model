from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from functions.main import cycle_until_balanced

# === CONFIGURATION ===
material = "CeO2"
T = 1073                # Temperature [K]
p = 100000              # Pressure [Pa]
n_CO2 = 1.0             # Moles of CO2 per cycle
n_H2 = 1.5              # Moles of H2 per cycle
n_oxide = 40            # Moles of oxide in the bed
x_CO2_0 = 0.998         # Initial mole fraction CO2
x_H2O_0 = 0.002         # Initial mole fraction H2O
mesh_range = np.arange(2, 250, 2)
flow_sequence = ((False, True), (True, False))

# === Initialize storage lists ===
solve_time = []
X_CO2_values = []
X_H2_values = []
O_bal_gas_values = []
O_bal_oxide_values = []
O_bal_oxide_red_values = []

# === Mesh Sweep Simulation ===
for mesh in mesh_range:
    tic = time.perf_counter()
    result = cycle_until_balanced(
        material=material,
        T=T,
        n_CO2=n_CO2,
        n_H2=n_H2,
        n_oxide=n_oxide,
        x_CO2_0=x_CO2_0,
        x_H2O_0=x_H2O_0,
        oxide_mesh=mesh,
        gas_mesh=mesh,
        flow_sequence=flow_sequence,
        return_details=True,
    )

    sequence_results = result["sequence_results"]
    n_CO = sum(cycle["n_CO"] for cycle in sequence_results)
    n_H2O = sum(cycle["n_H2O"] for cycle in sequence_results)
    n_oxide_reduced = sum(
        np.mean(cycle["delta_red"][-1] - cycle["delta_red"][0]) * n_oxide
        for cycle in sequence_results
    )
    n_oxide_oxidized = sum(
        np.mean(cycle["delta_red"][-1] - cycle["delta_ox"][-1]) * n_oxide
        for cycle in sequence_results
    )

    sequence_length = len(sequence_results)
    X_CO2 = n_CO / (n_CO2 * sequence_length)
    X_H2 = n_H2O / (n_H2 * sequence_length)
    O_bal_gas = n_CO / n_H2O
    O_bal_oxide = n_CO / n_oxide_oxidized
    O_bal_oxide_red = n_H2O / n_oxide_reduced

    X_CO2_values.append(X_CO2)
    X_H2_values.append(X_H2)
    O_bal_gas_values.append(O_bal_gas)
    O_bal_oxide_values.append(O_bal_oxide)
    O_bal_oxide_red_values.append(O_bal_oxide_red)

    toc = time.perf_counter()
    solve_time.append(toc-tic)
    print(f"mesh: {mesh}, cycles: {result['cycles']}, X_H2: {X_H2:.3f}, X_CO2: {X_CO2:.3f}, "
          f"MB_gas: {O_bal_gas:.3e}, MB_oxide: {O_bal_oxide:.3e}, t: {toc - tic:.2f}s")

# === Plot Results ===
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(3.4, 3.3), facecolor='white')
plt.plot(mesh_range, X_CO2_values, label="$X_{CO2}$", color = 'black')
plt.plot(mesh_range, X_H2_values, label="$X_{H2}$", color = 'grey')
plt.xlabel("$N_\\mathrm{oxide}$, $N_\\mathrm{gas}$  [-]")
plt.ylabel("Conversion extent $X_i$ [-]")
plt.axvline(x = 100, color = 'black', ls = '--', label = 'default value')
plt.legend()

plt.xlim(0, 250)
filename = Path("plots") / "mesh_test_conversion.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.figure()
plt.plot(mesh_range, solve_time, label="$X_{CO2}$")
plt.xlabel("Number of mesh elements [-]")
plt.ylabel("Solve time [-]")
filename = Path("plots") / "mesh_test_solve_time.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.figure()
plt.plot(mesh_range, O_bal_gas_values, label="O-balance CO/H$_2$O")
plt.plot(mesh_range, O_bal_oxide_values, label="O-balance oxide/CO$_2$")
plt.plot(mesh_range, O_bal_oxide_red_values, label="O-balance H$_2$O/oxide")
plt.xlabel("Mesh Size")
plt.ylabel("Oxygen Balance Error []")
plt.legend()
filename = Path("plots") / "mesh_test_mass_balance.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
