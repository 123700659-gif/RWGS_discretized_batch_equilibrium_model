"""
heat_map_model.py

Cantera equilibrium reference:
   A single-panel TP-equilibrium map (no reactor, no flow arrangement).
   Used for validation and to bound the reactor result.

Figures and data are written to
plots/ and data/
"""

import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

import cantera as ct

from functions.main import cycle_until_balanced
from functions.batch_equilibrium_solver import EquilibriumDomainError

GAS_MECH = "gri30.yaml"  

# sequences arrangements

FLOW_SEQUENCES = {
    "Alternating countercurrent": [(False, True), (True, False)],
    "Countercurrent":             [(False, True)],
    "Cocurrent":                  [(False, False)],
}

# sweep only varies the ratio and T
PHASE_BASE = dict(
    material="CeZr20",
    n_oxide=10,
    x_H2O_0=0.005,
    x_CO2_0=0.998,
    oxide_mesh=100,
    gas_mesh=100,
    O_balance_tolerance=1e-3,
    delta_tolerance=1e-4,
)


def run_cycle(T, p, n_CO2, n_H2, flow_configuration, gas, return_details=False):
    """
    Run the repository's discretized reactor/cycle model for one operating
    point and return CO2 conversion.

    Parameters
    ----------
    T : float            reactor temperature [K]
    p : float            pressure [Pa]  (passed through; the repository model is isobaric so p affects only Cantera thermo)
    n_CO2, n_H2 : float  inlet molar amounts [mol] per oxidation/reduction phase
    flow_configuration : str
        "Alternating countercurrent" | "Countercurrent" | "Cocurrent"
    gas : ct.Solution    Cantera gas object (thermo backend)

    Returns
    -------
    X_CO2 : float
        CO2 conversion at the oxidation outlet, defined as
            X_CO2 = (x_CO2_in - x_CO2_out) / x_CO2_in
        which is the *feed conversion* of the limiting reactant and reduces
        to n_CO/(n_CO + n_CO2_out) only if no other carbon species forms.
    If return_details is True, return the conversion and the initial solid-
    domain diagnostic instead of the scalar conversion.
    """
    if flow_configuration not in FLOW_SEQUENCES:
        raise ValueError(f"Unknown flow configuration: {flow_configuration!r}")

    # make sure the Cantera gas is at the requested state before the model touches it (the repository model uses it for thermo lookups)
    gas.TP = T, p

    result = cycle_until_balanced(
        T=T, n_CO2=n_CO2, n_H2=n_H2,
        flow_sequence=FLOW_SEQUENCES[flow_configuration],
        return_details=True,
        **PHASE_BASE,
    )

    last_ox = result["sequence_results"][-1]
    x_CO2_out_grid = last_ox["x_CO2_ox"]          # shape (gas_mesh, n_pairs)
    outlet_ox = 0 if last_ox["left_flow_ox"] else -1
    x_CO2_out = float(x_CO2_out_grid[:, outlet_ox].mean())

    # Feed conversion of CO2 (limiting carbon reactant)
    conversion = 1.0 - x_CO2_out / PHASE_BASE["x_CO2_0"]
    if return_details:
        return {
            "conversion": conversion,
            "initial_equilibrium_diagnostic": result[
                "initial_equilibrium_diagnostic"
            ],
        }
    return conversion


# Operating axes

nH2_range = np.arange(1.0, 4.0 + 1e-9, 0.05)   # H2/CO2 ratio 1.0 -> 4.0
T_range   = np.arange(700, 1600 + 1e-9, 10)    # 700 -> 1690 K
p         = 100_000.0
n_CO2     = 1.0

CONFIGS = list(FLOW_SEQUENCES.keys())

T_profile = 1200
ratio_profile = 2.0

fig, (ax_ox, ax_red) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for name, flow_sequence in FLOW_SEQUENCES.items():
    result = cycle_until_balanced(
        T=T_profile,
        n_CO2=n_CO2,
        n_H2=ratio_profile * n_CO2,
        flow_sequence=flow_sequence,
        return_details=True,
        **PHASE_BASE,
    )

    last_cycle = result["sequence_results"][-1]
    delta_ox_end = last_cycle["delta_ox"][-1, :]
    delta_red_end = last_cycle["delta_red"][-1, :]
    x = np.linspace(0, 1, delta_ox_end.size)

    ax_ox.plot(x, delta_ox_end, label=name)
    ax_red.plot(x, delta_red_end, label=name)

ax_ox.set_title("End of oxidation")
ax_red.set_title("End of reduction")
for ax in (ax_ox, ax_red):
    ax.set_xlabel("Normalized reactor position, x")
    ax.legend()

ax_ox.set_ylabel(r"Final $\delta$")
fig.tight_layout()
plt.show()

# Cantera gas 

gas = ct.Solution(GAS_MECH)

X_reactor = {cfg: np.full((len(T_range), len(nH2_range)), np.nan) for cfg in CONFIGS}
DOMAIN_STATUS = {
    "valid": 0,
    "initial_equilibrium_outside_solid_domain": 1,
    "batch_equilibrium_outside_modeled_domain": 2,
    "other_simulation_error": 3,
    "initial_equilibrium_and_batch_outside_modeled_domain": 4,
}
X_reactor_domain_status = {
    cfg: np.full((len(T_range), len(nH2_range)),
                 DOMAIN_STATUS["other_simulation_error"], dtype=np.uint8)
    for cfg in CONFIGS
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # non-convergence is handled below
    for i_T, T in enumerate(T_range):
        for i_r, ratio in enumerate(nH2_range):
            n_H2 = ratio * n_CO2
            for cfg in CONFIGS:
                try:
                    result = run_cycle(
                        T, p, n_CO2, n_H2, cfg, gas, return_details=True
                    )
                    X_reactor[cfg][i_T, i_r] = result["conversion"]
                    diagnostic = result["initial_equilibrium_diagnostic"]
                    if diagnostic and diagnostic["outside_solid_domain"]:
                        X_reactor_domain_status[cfg][i_T, i_r] = DOMAIN_STATUS[
                            "initial_equilibrium_outside_solid_domain"
                        ]
                    else:
                        X_reactor_domain_status[cfg][i_T, i_r] = DOMAIN_STATUS["valid"]
                except EquilibriumDomainError as exc:
                    diagnostic = exc.initial_equilibrium_diagnostic
                    status = (
                        "initial_equilibrium_and_batch_outside_modeled_domain"
                        if diagnostic and diagnostic["outside_solid_domain"]
                        else "batch_equilibrium_outside_modeled_domain"
                    )
                    X_reactor_domain_status[cfg][i_T, i_r] = DOMAIN_STATUS[status]
                    print(
                        f"[outside model domain] {cfg} T={T:.0f} "
                        f"ratio={ratio:.2f}: {exc}"
                    )
                except Exception as exc:
                    print(f"[warn] {cfg} T={T:.0f} ratio={ratio:.2f}: {exc}")
                    X_reactor[cfg][i_T, i_r] = np.nan

for cfg in CONFIGS:
    clamped_points = np.count_nonzero(
        X_reactor_domain_status[cfg]
        == DOMAIN_STATUS["initial_equilibrium_outside_solid_domain"]
    )
    if clamped_points:
        print(
            f"[domain summary] {cfg}: initial feed/solid equilibrium was "
            f"clamped to a modeled endpoint at {clamped_points} operating points; "
            "see X_reactor_domain_status in data/heat_map_data.npz"
        )

# Cantera equilibrium reference (single map, no reactor)

X_equil = np.full((len(T_range), len(nH2_range)), np.nan)
for i_T, T in enumerate(T_range):
    for i_r, ratio in enumerate(nH2_range):
        n_H2 = ratio * n_CO2
        total = n_H2 + n_CO2
        gas.TPX = T, p, {"H2": n_H2 / total, "CO2": n_CO2 / total}
        gas.equilibrate("TP")
        xCO2 = gas.X[gas.species_index("CO2")]
        X_equil[i_T, i_r] = 1.0 - xCO2 * total / n_CO2   # carbon-feed conversion

T_grid, ratio_grid = np.meshgrid(T_range, nH2_range, indexing="ij")

# Shared colour scale over reactor panels only

def finite_limits(*maps):
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in maps])
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())

vmin_reactor, vmax_reactor = finite_limits(*X_reactor.values())

levels_contour = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Save data

np.savez_compressed(
    "data/heat_map_data.npz",
    T_range=T_range, nH2_range=nH2_range, p=p,
    X_equil=X_equil,
    **{f"X_reactor_{i}": X_reactor[cfg] for i, cfg in enumerate(CONFIGS)},
    **{f"X_reactor_domain_status_{i}": X_reactor_domain_status[cfg]
       for i, cfg in enumerate(CONFIGS)},
)
with open("data/heat_map_meta.json", "w") as f:
    json.dump({
        "p_Pa": p,
        "n_CO2_mol": n_CO2,
        "configs": CONFIGS,
        "flow_sequences": {k: [list(p) for p in v] for k, v in FLOW_SEQUENCES.items()},
        "reactor_vmin": vmin_reactor, "reactor_vmax": vmax_reactor,
        "domain_status_codes": DOMAIN_STATUS,
        "mechanism": GAS_MECH,
    }, f, indent=2)



if __name__ == "__main__":
    print("\nWrote:")
    print("  plots/heat_map_flow_arrangements.{png,pdf}")
    print("  plots/heat_map_equilibrium_reference.{png,pdf}")
    print("  data/heat_map_data.npz")
    print("  data/heat_map_meta.json")