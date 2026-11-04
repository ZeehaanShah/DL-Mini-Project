# NASA Irradiance Constraint Analysis — Phase 3

Frozen checkpoint `EXP_NASA_DL_W1`; NASA POWER Mumbai 2023; target `ALLSKY_SFC_SW_DWN`, hourly accumulated shortwave energy in Wh/m²; one-hour-ahead from a 24-hour input. No training.

Feature order is `ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M`. Input/target standardization uses training-only fitted scalers; inverse target scaling returns Wh/m². No target, feature order, or unit mismatch was found. The scalar linear output head has no nonnegative constraint; inverse scaling can yield negative irradiance. Negatives concentrate at night/twilight. The solar conversion rejects materially negative input; frontend nonnegative projection is the physical guard. Projection is not proof of model quality.

| Split | N | Raw MAE/RMSE/R2 | Projected MAE/RMSE/R2 | Negative raw |
|---|---:|---|---|---:|
| Validation | 1,290 | 16.9405/30.5673/0.9870 | 16.3367/30.5070/0.9871 | 261 (20.23%) |
| Test | 1,242 | 16.2575/30.1767/0.9844 | 15.4668/30.1125/0.9845 | 339 (27.29%) |

Descriptive daylight is actual target >50 Wh/m². Test daylight N=519: raw/projected 30.7740/44.9911/0.9519; none changed. <=50 N=723: raw 5.8370/10.5484/-8.5505; projected 4.4786/10.2285/-7.9799. Very low nighttime variance makes R2 unstable. Validation range 2023-09-15 12:00–2023-11-08 05:00; test 2023-11-10 06:00–2023-12-31 23:00.

Keep projection as downstream physical guard and raw/projected scores distinct. Preserve checkpoint. Any constrained head/target transform needs new experiment ID, validation-based selection and one held-out test comparison. All-hours R2 alone does not establish operational daylight performance.
