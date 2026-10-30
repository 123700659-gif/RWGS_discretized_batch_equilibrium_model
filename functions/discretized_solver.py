import numpy as np

from functions.batch_equilibrium_solver import batch_equilibrium


def sequential_batch_equilibrium(
    mu_O_delta_func, mu_O_gas_func, x_0,
    delta_t_x, x_gas_t_x, d_delta, d_X, mbf,
    left_flow=False,
):
    """Equilibrate successive gas batches with a spatially discretized solid.

    All gas composition variables x refer to the oxidiser mole fraction:
    H2O in a binary H2O/H2 mixture or CO2 in a binary CO2/CO mixture.
    The remaining fraction, 1 - x, is H2 or CO, respectively.

    Args:
        mu_O_delta_func (callable): Solid oxygen chemical potential versus delta.
        mu_O_gas_func (callable): Gas oxygen chemical potential versus H2O or CO2 fraction.
        x_0 (float): Inlet oxidiser mole fraction (H2O or CO2) for every gas batch.
        delta_t_x (ndarray): Solid grid, shape (gas_mesh, oxide_mesh). On entry,
            row zero contains the initial delta profile. Updated in place.
        x_gas_t_x (ndarray): Oxidiser mole fraction grid with the same shape,
            updated in place. The reductant fraction grid is 1 - x_gas_t_x.
        d_delta (ndarray): Nonnegative delta shifts to probe.
        d_X (ndarray): Corresponding gas shifts, equal to mbf * d_delta.
        mbf (float): Solid-to-gas mole ratio for one contacting pair of elements.
        left_flow (bool): True for flow from the last column toward column zero.
            Flip the initial solid profile before solving and both grids back
            afterward. Use equal settings for cocurrent phases and opposite
            settings for countercurrent phases.

    Returns:
        tuple: Updated solid and gas grids, both in the original spatial
            orientation. Each cell records the state after its batch contact.

    No conversion stopping conditions are applied. Equilibrium-search errors
    propagate from batch_equilibrium.
    """
    if delta_t_x.ndim != 2 or x_gas_t_x.shape != delta_t_x.shape:
        raise ValueError("Solid and gas grids must have the same two-dimensional shape.")

    gas_mesh, oxide_mesh = delta_t_x.shape
    if gas_mesh == 0 or oxide_mesh == 0:
        raise ValueError("Solid and gas grids must be nonempty.")

    if left_flow:
        delta_t_x[0] = np.flip(delta_t_x[0])

    initial_delta = delta_t_x[0].copy()

    for t_step in range(gas_mesh):
        for x_step in range(oxide_mesh):
            delta_i = initial_delta[x_step] if t_step == 0 else delta_t_x[t_step - 1, x_step]
            x_i = x_0 if x_step == 0 else x_gas_t_x[t_step, x_step - 1]

            d_d = batch_equilibrium(
                mu_O_delta_func, mu_O_gas_func,
                delta_i, x_i, d_delta, d_X,
            )
            delta_t_x[t_step, x_step] = delta_i + d_d[0]
            x_gas_t_x[t_step, x_step] = x_i + mbf * d_d[0]

    if left_flow:
        delta_t_x[:] = np.flip(delta_t_x, axis=1)
        x_gas_t_x[:] = np.flip(x_gas_t_x, axis=1)

    return delta_t_x, x_gas_t_x
