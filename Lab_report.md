# Chemistry Final Year Project - Laboratory Report

## Summer Internship
The idea was to be able to perform the RWGS with counter and co-current from [source code](https://github.com/bulfinb/RWGS_disretized_batch_equilibrium_model.git).

The main issue was to be able to perform the discretized oxidation in both directions.

Once both examples have been up to date and working, the mass\_balance\_test.py file has shown irregular behaviour.


## Week 1 - 07/09

Work on mass\_balance.py file:

```bash
counter_current = True: 

Final CO2 conversion: 0.013045005693614664
```

2nd plot, no reduction or oxidation

```bash
counter_current = False:

output = "File "/Users/.../mass_balance_test.py", line 61, in <module>
delta_ox_prev = delta_t_x_ox[-1]
                ^^^^^^^^^^^^
NameError: name 'delta_t_x_ox' is not defined. Did you mean: 'delta_x_0'?
```
by substituting with 

```bash
delta_ox_prev = delta_x_0[0]
```
output:
```bash
Final CO2 conversion: 1.3851142455223453e-12
```
  
Output:

```bash
File "/Users/mathildedeclercq/untitled folder/untitled folder #2/RWGS_disretized_batch_equilibrium_model/model_testing/mass_balance_test.py", line 71, in <module>
    delta_x_0=delta_ox_prev,   # use previous oxidation profile
              ^^^^^^^^^^^^^
NameError: name 'delta_ox_prev' is not defined. Did you mean: 'delta_ox_end'?
```

Fixed the `NameError` in `mass_balance_test.py`.

Changes:
- Initialize `delta_ox_prev` from the current `delta_x_0` before calling `simulate_cycle`.
- Removed the late assignment that occurred after the simulation.
- Preserved the previous oxidation profile for both the next cycle and reduction mass-balance calculation.

Validation:
- `python -m py_compile model_testing/mass_balance_test.py` succeeds.
- The script now proceeds beyond the original `NameError`; the full run was stopped because it remained in the plotting/runtime environment.

```bash
counter_current = True

Final CO2 conversion: 0.19897191427374794

counter_current = False

Final CO2 conversion: 0.03480131730274705
````

Results : high oscillation, irregularities 

## Week 1 - 14/09
### Monday:
-   Try `mass_balance_test.py` with the old `mass_balance.py`, and results look weird $\rightarrow$ possible that `mass_balance.py` isn't the issue. 
-   When looking at `discretized_oxidation.py`, the output for example 1 is incorrect.
-   `discretized_oxidation_co_current.py` has been fixed and now `mass_balance_test.py` gives correct output for plot 1 for:
```bash 
counter_current = False
``` 
but not for the 2nd plot $\rightarrow$ reduction decreases.
```C++
if d_d.size > 0 else 0.0
````
in `discretized_reduction.py` could be the issue.
-   The correct fix is to establish and consistently use the physical reduction outlet for `counter_current=True`, then retain the near-zero guard for genuinely zero $H_{2}$ conversion.

Changes:

-   In `discretized_reduction.py`, documented that after restoring the physical orientation, the gas outlet is column 0 for counter-current flow.
-   In `mass_balance.py`:
Counter-current reduction reads 
```
x_H2O_t_x_red.T[0]
```

Co-current reduction continues to read 
```
x_H2O_t_x_red.T[-1]
```
-   Both X_H2 and n_H2O use the same physical outlet.
-   Added a near-zero guard for the gas oxygen balance in `mass_balance.py`, returning NaN instead of a huge numerical artifact when H₂O production is effectively zero.
-   Updated `mass_balance.py` to select the correct outlet based on `counter_current`.
-   The spatial arrays are reversed relative to one another, as expected from the different flow orientations, but the physical conversions and balances now agree. 

### Tuesday:
-   `discretization_tests.py` doesn't balance.

```bash
File "/Users/mathildedeclercq/untitled folder/untitled folder #2/RWGS_disretized_batch_equilibrium_model/model_testing/discretization_tests.py", line 79, in <module>
    plt.set_xlim(0, 250)
    ^^^^^^^^^^^^
AttributeError: module 'matplotlib.pyplot' has no attribute 'set_xlim'
````
Fix:
```python
plt.xlim(0, 250)
````

To do:
-   2nd plot of `mass_balance_test.py` still looks weird
-   Try `mu_O_materials_comparison.py`

#### Meeting 16/09
default = right hand flow

counter_current
reduction left= False
oxidation left=True

co-current:
left = False

simulate cycle
one = co_current
another one alternating



left = True
remove any no intersection exception
add reduction

$\rightarrow$ do a git repository, change `counter_current`to `left`, with right being default setting.

-   first file: `batch_equilibrium_solver`:
```python
    diff_0 = mu_O_H2O_func(x_H2O_i) - mu_O_delta_func(delta_i)

    if diff_0 < 0:
        print("diff_0 < 0")
    else:
        print("diff_0 > 0")
````
for left = False:
```bash
diff_0 < 0
````
orginal plots have been saved under `try1.*.png`

```python
diff_0 = mu_O_CO2_func(x_CO2_i) - mu_O_delta_func(delta_i)

    #debug
    if diff_0 < 0:
        print("diff_0 < 0")
    else:
        print("diff_0 > 0")
````
for left = False:
```bash
diff_0 > 0
````
-   `discretized_oxidation.py``

`oxidation_x0_bc`:

```python
    for t_step in range(1, gas_mesh):

        d_d = batch_equilibrium_solver.oxidation(mu_O_delta_func, mu_O_CO2_func,
                                                                delta_t_x0[t_step - 1], x_CO2_0, d_delta, d_X)
                    
        delta_t_x0[t_step] = delta_t_x0[t_step - 1] - d_d[0]
        x_CO2_t_x0[t_step] = x_CO2_0 - d_d[0] * mbf_ox
        if left:
            # Guard against solver overshoot outside physical bounds
            delta_t_x0[t_step] = np.clip(delta_t_x0[t_step], delta_min_oxidation, delta_t_x0[t_step - 1])
            x_CO2_t_x0[t_step] = np.clip(x_CO2_t_x0[t_step], 0.0, 1.0)
    
    return delta_t_x0, x_CO2_t_x0
````

-    `discretized_reduction`:

```python
left = True 
````

-   `mass_balance.py`:

Should I change : ?
```python
# Change in delta across cycle (reduction to oxidation)
    d_delta_CeO2 = delta_t_x_red[-1] - delta_t_x_ox[-1]
    nO_CeO2 = d_delta_CeO2.sum() * n_oxide / len(delta_t_x_red[-1])
````

unchanged files:
-   `build_arrays.py`
-   `build_mu_O_functions.py`
-   `energy_balanced.py`
-   `gas_thermodynamics.py``
-   `oxide_thermodynamics`

current issue:
-   I think both plots for 
````
reduction=oxidation=True
````
are wrong. The issue is there.