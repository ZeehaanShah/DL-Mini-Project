# Baseline Comparison Report — Phase 3

Date: 2026-10-27. Frozen checkpoints only; no training. Baselines use the exact model 24-hour input history, same targets/timestamps and physical units. No baseline parameters were fit to test targets.

| Model/baseline | Test period | N | MAE | RMSE | R2 | Unit | Finding |
|---|---|---:|---:|---:|---:|---|---|
| NASA W1 raw model | 2023-11-10 06:00–2023-12-31 23:00 | 1,242 | 16.2575 | 30.1767 | 0.9844 | Wh/m2 | Beats same-window baselines. |
| NASA W1 projected | same | 1,242 | 15.4668 | 30.1125 | 0.9845 | Wh/m2 | Clips 339/1,242 negatives (27.29%). |
| NASA persistence (last input) | same | 1,242 | 56.5286 | 90.9381 | 0.8582 | Wh/m2 | |
| NASA daily seasonal (first input) | same | 1,242 | 17.5779 | 44.8219 | 0.9656 | Wh/m2 | |
| NASA 24-hour mean | same | 1,242 | 209.0677 | 241.2847 | 0.0019 | Wh/m2 | |
| Wind C2 model | 2014-11-10 06:30–2014-12-31 23:30 | 1,242 | 0.5908 | 0.8216 | 0.7576 | m/s | |
| Wind persistence | same, last input | 1,242 | 0.6002 | 0.8679 | 0.7295 | m/s | C2 modestly wins. |
| Wind seasonal (first input) | same | 1,242 | 1.2661 | 1.6807 | -0.0142 | m/s | |
| Demand D2 model | 2017-09-16 07:00–2017-12-31 23:00 | 2,561 | 2.6152 | 4.0673 | 0.9234 | kWh | |
| Demand persistence (last input) | same | 2,561 | 2.5336 | 4.0823 | 0.9228 | kWh | D2 worse MAE; slightly better RMSE/R2. |
| Demand seasonal (first input) | same | 2,561 | 6.7775 | 9.9977 | 0.5369 | kWh | |
| Demand 24-hour mean | same | 2,561 | 5.6184 | 7.4499 | 0.7429 | kWh | |

NASA daylight descriptive split uses actual >50 Wh/m2: test daylight N=519 model 30.7740/44.9911/0.9519; persistence 123.139/136.457/0.5579; daily seasonal 41.819/69.290/0.8860 (MAE/RMSE/R2). <=50 Wh/m2 N=723 raw 5.8370/10.5484/-8.5505, projected 4.4786/10.2285/-7.9799. Night R2 is unstable at low target variance. Validation all-hours raw/projected N=1,290: 16.9405/30.5673/0.9870 and 16.3367/30.5070/0.9871; 261 negatives (20.23%).

**Correction:** old Phase 2 NASA baselines near MAE=172.28/RMSE=296.07 are withdrawn. This audit recomputes aligned baselines from the exact 24-hour model input window and t+1 target; the prior cause is not established. All-hours NASA includes night records. Model E is not a synchronized real-site series. Weekly-naive unavailable under 24-hour model-input contract.
