# Benchmark Summary

| config | optimiser | backend | integrator | continuity(w) | completed | best_loss | duration[s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| config_benchmark.yaml | optuna | numpy | rk4 | True (0.0) | 300 | 2.720e-15 | 4.5 |
| config_benchmark_euler_numpy.yaml | optuna | numpy | euler | True (0.0) | 300 | 4.971e-16 | 2.7 |
| config_benchmark_rk23_numpy.yaml | optuna | numpy | rk23 | True (0.0) | 300 | 6.771e-16 | 4.7 |
| config_benchmark_rk4_jax.yaml | optuna | jax | rk4 | True (0.0) | 300 | 5.690e-16 | 33.6 |
| config_benchmark_constrained.yaml | optuna | numpy | rk4 | True (0.01) | 300 | 7.361e-16 | 7.4 |
| config_benchmark_grad.yaml | gradient | numpy | rk4 | True (0.0) | 1 | 1.702e-14 | 0.2 |
