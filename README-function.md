Copyright contributors to the shortest-path-mechanism project


# Dimension of an environment
- n: # of agents
- m: # of options
- d: size of every type domain

# Histogram
```
hist_diff(fix, n_max, m_max, d_max, value_min, value_max, reps)
```
- fix: choose the parameter to fix from ["n", "m", "d"]
- n: fixed when fix="n"
- m: iterate over [1, m_max] when fix="n"
- d: random in [1, d_max] when fix="n"
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

# Plot

## Difference from VCG-budget

### Vs. n = |N|
```
plot_diff_vs_n(n_max, m_max, d_max, value_min, value_max, reps, errfn)
```
- n: iterate over [1, n_max]
- m: random in [1, m_max]
- d: random in [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

### Vs. m = |X|
```
plot_diff_vs_m(n, m_max, d_max, value_min, value_max, reps, errfn)
```
- n: fixed
- m: iterate over [1, m_max]
- d: random in [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

### Vs. d = |V_i|
```
plot_diff_vs_d(n, m_max, d_max, value_min, value_max, reps, errfn)
```
- n: fixed
- m: random in [1, m_max]
- d: iterate over [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

# The following functions are not used in the experiments shown in the paper

## Each of Proposed and VCG-budget

### Vs. n = |N|
```
plot_budget_vs_n(n_max, m_max, d_max, value_min, value_max, reps)
```
- n: iterate over [1, n_max]
- m: random in [1, m_max]
- d: random in [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

### Vs. m = |X|
```
plot_budget_vs_m(n, m_max, d_max, value_min, value_max, reps)
```
- n: fixed
- m: iterate over [1, m_max]
- d: random in [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used

### Vs. d = |V_i|
```
plot_budget_vs_d(n, m_max, d_max, value_min, value_max, reps)
```
- n: fixed
- m: random in [1, m_max]
- d: iterate over [1, d_max]
- v_i(x): random in [value_min, value_max]
- reps: # of instances used
