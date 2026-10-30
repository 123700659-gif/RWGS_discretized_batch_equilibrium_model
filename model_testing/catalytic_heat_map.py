# Catalytic RWGS equilibrium heat map

import os

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

# Feed ratio scan for catalytic RWGS equilibrium
nH2_range = np.arange(1.0, 4.0, 0.05)  # H2/CO2 feed ratio [-]
T_range = np.arange(573, 1700, 10)      # Temperature [K]
p = 100_000.0                           # Pressure [Pa]

# Create the gas phase model
gas = ct.Solution("gri30.yaml")

# Store equilibrium CO2 conversion map
X_CO2_map = np.zeros((len(nH2_range), len(T_range)))

for i, n_H2 in enumerate(nH2_range):
    total = n_H2 + 1.0
    x_H2 = n_H2 / total
    x_CO2 = 1.0 / total

    for j, T in enumerate(T_range):
        gas.TPX = T, p, {"H2": x_H2, "CO2": x_CO2}
        gas.equilibrate("TP")

        x_CO = gas.X[gas.species_index("CO")]
        x_CO2_eq = gas.X[gas.species_index("CO2")]
        X_CO2_map[i, j] = x_CO / (x_CO + x_CO2_eq)

T_grid, nH2_grid = np.meshgrid(T_range, nH2_range)

# Plot
fig = plt.figure(figsize=(4.2, 3.2), facecolor="white")
ax0 = plt.subplot()
ax0.set_xlabel("$T$ [K]")
ax0.set_ylabel("$\mathrm{H_2/CO_2}$ [-]")

im = ax0.pcolormesh(T_grid, nH2_grid, X_CO2_map, rasterized=True, vmin=0.1, vmax=1.0)
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ct_lines = ax0.contour(T_grid, nH2_grid, X_CO2_map, levels=levels, colors="black")
ax0.clabel(ct_lines, fmt="%1.1f")
cb = fig.colorbar(im, ax=ax0)
cb.set_label(label="$X_\mathrm{CO_2}$")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "figure_1c.png"), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join("plots", "figure_1c.pdf"), bbox_inches="tight")
plt.show()