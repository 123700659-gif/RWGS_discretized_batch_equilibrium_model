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

#### Thursday:

```python
        reduction: 
            -left = True: original
        oxidation:
            -left = True: original
````
-   `example_1_simulate_cycle` has been fixed and outputs are coherent
-   `mass_balance_test.py`has been ran once, results for original settings are incoherent $\rightarrow$ to fix

#### Friday:

-   `example_1_simulate_cycle.py` :reduction arrow was pointing the wrong direction :
```python
reduction = True r'H$_2$ flow $\\rightarrow$'
oxidation = True r'$\\leftarrow$ CO$_2$ flow'
````
-   `mass_balance.py`: `d_delta_*` name changed to fit the material used
-   `mass_balance.py`: calculation assumes 1:1 reaction
-   `main.py`: MAIN ISSUE in previous code $\rightarrow$ deleted now:
```python
elif reduction and not oxidation:
        # Use reduction output as initial condition for oxidation
        delta_t_x_ox[0] = delta_t_x_red[-1]
        delta_t_x_ox[0] = np.flip(delta_t_x_ox[0])

        # Run oxidation simulation
        delta_t_x_ox, x_CO2_t_x_ox = compute_oxidation_step(
            mu_O_delta_func, mu_O_CO2_func, x_CO2_0, delta_min,
            delta_t_x_ox, x_CO2_t_x_ox, d_delta_ox, d_X_ox,
            mbf_ox, oxidation=oxidation, gas_mesh=gas_mesh, oxide_mesh=oxide_mesh
        )
        delta_t_x_ox = np.flip(delta_t_x_ox, axis=1)
````
To do: 
-   keep fixing `mass_balance_test.py`and `discretizations_test.py`


## Week 2 - 21/09

### Monday: 
Changes done:
-   `mass_balance_test.py` $\rightarrow$ `l.65`
-   `mass_balance.py` $\rightarrow$ add debug
-    on `main.py`:
```python
elif reduction and not oxidation:
        # Use reduction output as initial condition for oxidation
        delta_t_x_ox[0] = delta_t_x_red[-1]
        delta_t_x_ox[0] = np.flip(delta_t_x_ox[0])

        # Run oxidation simulation
        delta_t_x_ox, x_CO2_t_x_ox = compute_oxidation_step(
            mu_O_delta_func, mu_O_CO2_func, x_CO2_0, delta_min,
            delta_t_x_ox, x_CO2_t_x_ox, d_delta_ox, d_X_ox,
            mbf_ox, oxidation=oxidation, gas_mesh=gas_mesh, oxide_mesh=oxide_mesh
        )
        delta_t_x_ox = np.flip(delta_t_x_ox, axis=1)
````

This is the key point:
-   if in the code: `mass_balance_test.py` first plot comes out ok, 2nd one weird, but `example_1_simulate_cycle.py` 3rd plot is wrong.
-   if not in the code: both `mass_balance_test.py` plots are wrong but `example_1_simulate_cycle.py` is ok.
`example_1_simulate_cycle.py`: 1st plot never changes
$\rightarrow$ `discretized_reduction.py`changed: 
```python
if not oxidation:
        delta_x_t = np.flip(delta_x_t, axis=1)
```
$\rightarrow$ the python code (see above with `elif`) has been re-integrated in `main.py`.


In `mass_balance_test.py`:
```python
reduction=True
oxidation=False
```
```bash
Cycle 0: solve time 0.61 s,  1.1399222521231989 0.6577807127025062
Cycle 1: solve time 0.42 s,  1.100526066897315 0.5998404423056906
Cycle 2: solve time 0.43 s,  1.1255598544231764 0.6003822330810967
Cycle 3: solve time 0.43 s,  1.1668895066945502 0.5913077001612224
Cycle 4: solve time 0.44 s,  1.200781251247346 0.6008350446742673
Cycle 5: solve time 0.45 s,  1.2615542619203983 0.5605336849457493
Cycle 6: solve time 0.45 s,  1.3228072583076236 0.5674691521576138
Cycle 7: solve time 0.45 s,  1.3773226032084904 0.5749707575058658
Cycle 8: solve time 0.45 s,  1.4311021054876147 0.5797209161211284
Cycle 9: solve time 0.45 s,  1.489554702020823 0.5439469779883672
Cycle 10: solve time 0.45 s,  1.4973237442238196 0.5646849173349094
Cycle 11: solve time 0.46 s,  1.5019514016403523 0.5646849173349094
Cycle 12: solve time 0.46 s,  1.504830142607878 0.5646849173349094
Cycle 13: solve time 0.45 s,  1.5066822112772678 0.5646849173349094
Cycle 14: solve time 0.45 s,  1.507912068791415 0.5646849173349094
Cycle 15: solve time 0.47 s,  1.5087541647611948 0.5646849173349094
Cycle 16: solve time 0.49 s,  1.5093488120092993 0.5646849173349094
Cycle 17: solve time 0.46 s,  1.5097799163241254 0.5646849173349094
Cycle 18: solve time 0.45 s,  1.5101009405161947 0.5646849173349094
Cycle 19: solve time 0.45 s,  1.5103436682154479 0.5646849173349094
Final CO2 conversion: 0.9791607030631446
```
```python
reduction=True
oxidation=True
```
```bash
Cycle 0: solve time 0.64 s,  0.9999999999999547 0.6581657564510776
Cycle 1: solve time 0.66 s,  0.9999999999999998 0.8311487190808882
Cycle 2: solve time 0.68 s,  0.9999999999999999 0.9331817509858046
Cycle 3: solve time 0.66 s,  1.0000000000000004 0.9520834998988955
Cycle 4: solve time 0.72 s,  1.0 0.962828485552031
Cycle 5: solve time 0.68 s,  1.0000000000000002 0.9698449293637391
Cycle 6: solve time 0.68 s,  1.0000000000000002 0.9747739993762063
Cycle 7: solve time 0.69 s,  1.0000000000000004 0.9784141437944188
Cycle 8: solve time 0.69 s,  1.0000000000000002 0.9812036168155593
Cycle 9: solve time 0.68 s,  1.0000000000000002 0.9834056684524024
Cycle 10: solve time 0.67 s,  1.0000000000000002 0.9851855473781354
Cycle 11: solve time 0.73 s,  0.9999999999999999 0.9866518385228823
Cycle 12: solve time 0.67 s,  1.0000000000000004 0.9878804286286709
Cycle 13: solve time 0.67 s,  0.9999999999999998 0.98892371218233
Cycle 14: solve time 0.67 s,  1.0000000000000002 0.989820338699021
Cycle 15: solve time 0.67 s,  1.0000000000000002 0.9905988882867952
Cycle 16: solve time 0.67 s,  1.0 0.9912805730234865
Cycle 17: solve time 0.67 s,  1.0 0.9918827274008992
Cycle 18: solve time 0.67 s,  1.0 0.9924181991283212
Cycle 19: solve time 0.68 s,  1.0 0.9928972323265887
Final CO2 conversion: 0.9903496071329413
```

Changes made in `mass_balance.py` : both plots look more coherent for 
```python
reduction=oxidation=True
```
even though 2nd plot has oxidation slightly decreasing, but for `oxidation=False`the output is clearly wrong. The oxide and gas values are going further apoart instead of closer.

### Tuesday:

-    Running of `discretization_tests.py`: no difference in the results beteween `oxidation=True` and `oxidation=False` $\rightarrow$ high irregularities, plots incoherent $\rightarrow$ `cycle_until_balanced`could be wrong
-   Running of `example_2`:
```python
reduction=True
oxidation=True
```
```bash
# of cycles till balanced= 16 | O balance oxide = 0.989957 | CO/H2O = 0.98908 
X_CO2 = 0.9799 | X_H2 = 0.6604 | Q_red = 120.602 kJ/mol | Q_ox = -86.763 kJ/mol
```
```python
reduction=True
oxidation=False
```
```bash
 # of cycles till balanced= 50 | O balance oxide = 0.956827 | CO/H2O = 0.32866 
X_CO2 = 0.5959 | X_H2 = 1.2087 | Q_red = 71.275 kJ/mol | Q_ox = -50.697 kJ/mol
```
Still an *issue with balancing oxide and gas values.*


