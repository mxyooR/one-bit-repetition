# One-bit repetition simulations

Code and recorded synthetic data for correlation estimation with noisy sign measurements.

```sh
python -m pip install -r requirements.txt
python experiments/verify_recorded_results.py
python experiments/plot_adaptive_results.py
```

The records contain 51 configurations, five methods, 2000 trials each, and
510000 estimates. The verifier reconstructs saved estimates and checks bit
budgets without new sampling. Plots and tables are written to `output/`.

To generate a separate run, copy `adaptive_model.py`,
`run_adaptive_validation.py`, and `adaptive_protocol.json` to a new directory,
then run `python run_adaptive_validation.py` there. Keep the recorded data.
The protocol fixes the seed and parameter grid. Set `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` for a single-thread run.
