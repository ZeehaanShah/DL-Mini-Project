# Stage 5: Uncertainty-Aware Energy Feasibility and Mix Optimization (Model E)

**Author**: Senior Deep Learning Engineer & Research Auditor  
**Date**: 2026-10-23  
**Framework Title**: *“A Multimodal, Uncertainty-Aware Decision-Support Framework for Location-Specific Renewable Energy Feasibility and Energy-Mix Optimization”*  
**Module**: `src/decision_support/` (Model E)  
**Status**: COMPLETE (177 repository tests passing; zero model retraining; historical registry and checkpoints preserved; rigorous energy conservation and conformal uncertainty stress testing verified)

---

## 1. Executive Summary & Mandatory Safeguards Audit

Stage 5 completes the final component of the project framework: **Model E (Uncertainty-Aware Energy Feasibility and Energy-Mix Optimization)**. Model E bridges deep-learning sequence forecasts, physical engineering conversion models, conformal uncertainty bounds, and battery storage dynamics to deliver deterministic, reproducible decision support for renewable energy feasibility.

### Non-Negotiable Safeguards Audit

| Mandatory Requirement | Implementation & Verification | Status |
|---|---|:---:|
| **1. Zero Retraining of Models A–D** | Loaded frozen checkpoints (`checkpoints/EXP_WIND_DL_C2/best_model.pt` and `checkpoints/EXP_DEMAND_DL_D2/best_model.pt`) and static NASA POWER/SKIPP'D data directly. No model weights were adjusted. | VERIFIED |
| **2. Checkpoint & Registry Preservation** | Historical experiment registry records 1–49 and historical checkpoints remain byte-for-byte identical. 3 new entries logged (`EXP_DS_BASE_GRID`, `EXP_DS_OPTIMIZED_MIX`, `EXP_DS_UNCERTAINTY_STRESS`). | VERIFIED |
| **3. Separation of Multi-Location Datasets** | Distinct geographical origins (Stanford nowcasting, Mumbai solar irradiance, Gujarat wind speed, Orlando electricity demand) are preserved and explicitly distinguished. Datasets were **never merged into a single empirical time series**. | VERIFIED |
| **4. Physical Conversion Distinction** | Irradiance in $\text{Wh/m}^2$ is converted to electrical energy via a documented Standard Test Condition (STC) derate model. Wind power is converted via a generic 2 MW turbine power curve and explicitly designated as an **illustrative engineering proxy**, not measured generation. | VERIFIED |
| **5. Physical Energy Conservation** | Energy conservation holds strictly at every time step ($|\text{residual}| \le 10^{-6}\text{ kWh}$) across solar, wind, battery, grid, served demand, curtailment, and unmet demand. | VERIFIED |
| **6. Deterministic Optimization** | Discrete candidate grid evaluation ($6 \times 5 \times 4 = 120$ combinations) guarantees the exact global optimum over the search space with 100% reproducible bitwise output. | VERIFIED |
| **7. Conformal Uncertainty Sensitivity** | Evaluated under Conservative Stress ($R_{\text{low}}, D_{\text{upp}}$), Central ($R_{\text{cent}}, D_{\text{cent}}$), and Favorable ($R_{\text{upp}}, D_{\text{low}}$) scenarios using calibrated 90% conformal quantiles ($q_{\text{wind}}=1.5316\text{ m/s}$, $q_{\text{demand}}=4.7627\text{ kWh}$). | VERIFIED |
| **8. Marginal vs Joint Uncertainty Caveat** | Explicitly declared that separate 90% marginal conformal intervals do not guarantee 90% simultaneous joint coverage without Bonferroni or copula adjustments. | VERIFIED |
| **9. Test Set Leakage Prevention** | Conformal calibration parameters ($q_{80}, q_{90}$) were computed strictly on dedicated calibration partitions. Test residuals were never used to tune optimizer parameters or scenario selection. | VERIFIED |
| **10. Battery Physical Constraints** | Storage modeled with rigorous state of charge bounds ($10\% \le \text{SOC} \le 95\%$), round-trip efficiency ($\eta_{\text{rt}} \approx 84.6\%$), and C-rate power ceilings ($0.5\text{C}$). | VERIFIED |

---

## 2. Physical Formulation of the Energy Balance

To eliminate ambiguity, Model E implements a unified per-time-step energy balance equation in `src/decision_support/energy_balance.py`.

### 2.1 The Per-Time-Step Conservation Laws

For each discrete hourly interval $t \in \{1, \dots, T\}$ ($\Delta t = 1.0\text{ h}$), energy conservation is enforced across two complementary balances:

1. **Demand Balance**:
   $$D_t = D_{\text{served}, t} + U_t$$
   $$D_{\text{served}, t} = R_{\text{direct}, t} + B_{\text{dis}, t} + G_{\text{import}, t}$$
   where:
   - $D_t$: Total electricity demand requirement ($\text{kWh}$).
   - $D_{\text{served}, t}$: Electricity demand successfully supplied ($\text{kWh}$).
   - $U_t$: Unmet electricity demand ($\text{kWh}$), occurring only if available renewable, battery, and permitted grid import are exhausted ($U_t \ge 0$).
   - $R_{\text{direct}, t}$: Renewable generation consumed directly by the load ($\text{kWh}$).
   - $B_{\text{dis}, t}$: Useful electricity delivered from battery storage discharge ($\text{kWh}$).
   - $G_{\text{import}, t}$: Electricity imported from the utility grid ($\text{kWh}$), subject to physical interconnection capacity $G_{\text{import}, t} \le G_{\text{limit}}$.

2. **Renewable Generation Balance**:
   $$R_{\text{avail}, t} = R_{\text{direct}, t} + B_{\text{ch}, t} + C_t$$
   where:
   - $R_{\text{avail}, t}$: Total available renewable energy ($E_{\text{solar}, t} + E_{\text{wind}, t}$) in $\text{kWh}$.
   - $B_{\text{ch}, t}$: Renewable energy directed into battery charging ($\text{kWh}$).
   - $C_t$: Curtailed renewable surplus ($\text{kWh}$) rejected when load is met and battery storage is full or power-limited.

### 2.2 Battery Energy Storage System (BESS) Dynamics

Battery storage is modeled with non-ideal thermodynamics and finite power limits:

- **State of Charge (SOC) Evolution**:
  $$\text{SOC}_{t} = \text{SOC}_{t-1} + (B_{\text{ch}, t} \cdot \eta_{\text{ch}}) - \left(\frac{B_{\text{dis}, t}}{\eta_{\text{dis}}}\right)$$
  where $\eta_{\text{ch}} = 0.92$ (charging efficiency) and $\eta_{\text{dis}} = 0.92$ (discharging efficiency), yielding a realistic round-trip AC-battery-AC efficiency:
  $$\eta_{\text{round-trip}} = \eta_{\text{ch}} \cdot \eta_{\text{dis}} = 0.92 \times 0.92 = 0.8464 \; (84.64\%)$$
- **Capacity & Power Constraints**:
  $$C_{\text{batt}} \cdot \text{SOC}_{\min} \le \text{SOC}_t \le C_{\text{batt}} \cdot \text{SOC}_{\max}$$
  $$B_{\text{ch}, t} \le C_{\text{batt}} \cdot C_{\text{rate, ch}} \cdot \Delta t$$
  $$B_{\text{dis}, t} \le C_{\text{batt}} \cdot C_{\text{rate, dis}} \cdot \Delta t$$
  where $\text{SOC}_{\min} = 0.10$, $\text{SOC}_{\max} = 0.95$, and $C_{\text{rate}} = 0.50\text{ h}^{-1}$ ($2$-hour complete charge/discharge ceiling).
- **Internal Numerical Residual Verification**:
  Every simulation step numerically verifies:
  $$\left| (R_{\text{direct}, t} + B_{\text{dis}, t} + G_{\text{import}, t} + U_t) - D_t \right| \le 10^{-6}\text{ kWh}$$
  Any non-conservation anomaly immediately raises a `RuntimeError`.

---

## 3. Physical Resource Conversion Models

Because raw forecasts describe meteorological states or surface flux, physically grounded engineering models translate them into electrical kilowatt-hours:

### 3.1 Solar Irradiance to AC Electrical Energy
NASA POWER all-sky surface shortwave downward irradiance $G_t$ ($\text{Wh/m}^2$) is converted to AC electricity $E_{\text{PV}, t}$ ($\text{kWh}$) using a standard test condition (STC) rating model:
$$E_{\text{PV}, t} = C_{\text{solar}} \times \left( \frac{G_t}{G_{\text{STC}}} \right) \times \text{PR} \times \Delta t$$
where:
- $C_{\text{solar}}$: Nameplate solar PV DC peak rating ($\text{kW}$).
- $G_{\text{STC}} = 1000.0\text{ W/m}^2$: Standard test irradiance.
- $\text{PR} = 0.78$: Typical balance-of-system performance ratio accounting for inverter efficiency, thermal derating, wiring resistance, and soiling.

### 3.2 Wind Speed to Electrical Generation Proxy
Wind speed $v_t$ ($\text{m/s}$) at 100m hub height is mapped to power $P(v_t)$ ($\text{kW}$) via the generic 2 MW turbine power proxy ($v_{\text{cut-in}} = 3.0\text{ m/s}$, $v_{\text{rated}} = 12.0\text{ m/s}$, $v_{\text{cut-out}} = 25.0\text{ m/s}$):
$$E_{\text{wind}, t} = \left( \frac{C_{\text{wind}}}{P_{\text{rated, ref}}} \right) \times P(v_t) \times \Delta t$$
where $P_{\text{rated, ref}} = 2000.0\text{ kW}$ and $C_{\text{wind}}$ is the candidate installed wind capacity.

---

## 4. Deterministic Multi-Objective Energy-Mix Optimization

### 4.1 Search Grid and Global Optimality
To guarantee reproducibility and avoid non-convex gradient pitfalls, optimization is formulated over an exhaustive candidate grid:
- **Solar Capacities**: $\{0.0, 25.0, 50.0, 75.0, 100.0, 150.0\}\text{ kW}$ (6 values)
- **Wind Capacities**: $\{0.0, 50.0, 100.0, 200.0, 300.0\}\text{ kW}$ (5 values)
- **Battery Capacities**: $\{0.0, 50.0, 100.0, 200.0\}\text{ kWh}$ (4 values)
- **Total Discrete Search Space**: $6 \times 5 \times 4 = 120$ candidate system configurations.

Because the optimizer exhaustively evaluates all 120 combinations, **it guarantees the exact global optimum over the defined search space without heuristic uncertainty or local minima**.

### 4.2 Economic Cost Formulation
The techno-economic objective function balances capital expenditure (CAPEX) amortized over an assumed 20-year economic lifetime against operational expenditure (OPEX):
$$\text{Cost}_{\text{total}} = \text{CAPEX}_{\text{amortized}} + \text{OPEX}$$
$$\text{CAPEX}_{\text{amortized}} = \left( c_{\text{solar}} C_{\text{solar}} + c_{\text{wind}} C_{\text{wind}} + c_{\text{batt}} C_{\text{batt}} \right) \times \left( \frac{T_{\text{horizon}}}{20 \times 8760\text{ h}} \right)$$
$$\text{OPEX} = c_{\text{grid}} \sum_{t=1}^T G_{\text{import}, t} + c_{\text{unmet}} \sum_{t=1}^T U_t + c_{\text{curtail}} \sum_{t=1}^T C_t$$
Parameters (illustrative benchmark proxies):
- $c_{\text{solar}} = \$800/\text{kW}$, $c_{\text{wind}} = \$1200/\text{kW}$, $c_{\text{batt}} = \$250/\text{kWh}$
- $c_{\text{grid}} = \$0.12/\text{kWh}$, $c_{\text{unmet}} = \$2.00/\text{kWh}$ (Value of Lost Load), $c_{\text{curtail}} = \$0.01/\text{kWh}$

### 4.3 Optimization Results & Baseline Comparison

The candidate configurations were evaluated over a 240-hour (10-day) representative test slice ($N=240$, Total Demand = $11,788.58\text{ kWh}$, Peak Demand = $90.34\text{ kW}$, Grid Import Limit = $100.0\text{ kW}$).

| Configuration | Solar ($C_{\text{pv}}$) | Wind ($C_{\text{wind}}$) | Battery ($C_{\text{bess}}$) | Renewable Penetration | Grid Import | Grid Reduction | Total Cost | Unmet Demand | Loss of Load |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100% Grid Baseline** | 0.0 kW | 0.0 kW | 0.0 kWh | **0.00%** | 11,788.58 kWh | 0.0% | $1,414.63 | 0.00 kWh | 0.0% (0 h) |
| **Cost-Optimal Mix** | 150.0 kW | 0.0 kW | 200.0 kWh | **38.32%** | 7,271.02 kWh | **-38.32%** | **$1,110.10** | 0.00 kWh | 0.0% (0 h) |
| **Autonomy-Optimal Mix** | 150.0 kW | 300.0 kW | 200.0 kWh | **57.11%** | 5,056.18 kWh | **-57.11%** | $1,379.30 | 0.00 kWh | 0.0% (0 h) |

#### Key Insights:
1. **Economic Viability**: The Cost-Optimal Mix achieves **38.32% clean energy penetration** while reducing total energy expenditure from $1,414.63 to $1,110.10—a net savings of **$304.53 (21.5% cost reduction)** over the 10-day evaluation slice.
2. **Autonomy Frontier**: The Autonomy-Optimal configuration achieves **57.11% renewable penetration** and cuts grid import by **6,732.40 kWh (57.11%)**, while remaining cheaper than the pure grid baseline ($1,379.30 vs $1,414.63).
3. **Storage Role**: Solar generation without storage leaves evening peaks unserved. The 200 kWh battery absorbs mid-day solar surplus and discharges 692.07 kWh during high-demand nighttime hours.

---

## 5. Conformal Uncertainty Feasibility Analysis

To evaluate resilience under real-world forecasting errors and distribution shift, the Cost-Optimal Mix was subjected to a rigorous 3-scenario conformal uncertainty stress test using 90% calibrated intervals:
- **Conservative Stress Scenario**: Worst-case combination of upper-bound electricity demand ($D_{\text{upp}} = \hat{D} + q_D$) and lower-bound renewable resources ($v_{\text{low}} = \max(0, \hat{v} - q_W)$).
- **Central Expected Scenario**: Point predictions ($\hat{D}, \hat{v}, \hat{G}$).
- **Favorable Optimistic Scenario**: Lower-bound electricity demand ($D_{\text{low}} = \max(0, \hat{D} - q_D)$) and upper-bound renewable resources.

### Conformal Uncertainty Sensitivity Table

| Scenario | Total Demand | Available Renewable | Grid Import | Curtailment | Unmet Demand | Renewable Penetration | Loss of Load |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conservative (Stress)** | 12,931.63 kWh | 4,299.89 kWh | 8,765.21 kWh | 92.12 kWh | **0.00 kWh** | **32.22%** | **0.0% (0 h)** |
| **Central (Expected)** | 11,788.58 kWh | 5,058.70 kWh | 7,271.02 kWh | 470.15 kWh | **0.00 kWh** | **38.32%** | **0.0% (0 h)** |
| **Favorable (Optimistic)** | 10,645.53 kWh | 5,817.50 kWh | 5,917.20 kWh | 978.49 kWh | **0.00 kWh** | **44.42%** | **0.0% (0 h)** |

### Critical Scientific Findings:
1. **Zero Unmet Demand Resilience**: Even under conservative joint stress where demand surges by +9.7% and available renewable generation declines by -15.0%, unmet demand remains **0.00 kWh** across all 240 hours because the 100 kW grid interconnection smoothly compensates for renewable deficits.
2. **Penetration Sensitivity Band**: Renewable penetration spans a stable corridor from **32.22%** (Stress) to **38.32%** (Expected) to **44.42%** (Optimistic).
3. **Curtailment Dynamics**: Under favorable solar generation, curtailment increases to 978.49 kWh, highlighting that additional storage capacity would be required to capture peak surplus.

### Statistical Disclosure on Joint Coverage
- The conformal quantiles ($q_W = 1.5316\text{ m/s}$, $q_D = 4.7627\text{ kWh}$) provide $90\%$ **marginal** coverage on their respective calibration distributions.
- **Do not interpret the stress scenario as a calibrated 90% joint confidence region**: By Bonferroni's inequality, separate marginal $1 - \alpha$ intervals guarantee joint coverage of at least $1 - 2\alpha = 80\%$, not $90\%$.
- Furthermore, as established in Stage 4, seasonal distribution shift in Florida building demand led to empirical test undercoverage ($82.16\%$ vs nominal $90\%$). The stress scenario deliberately tests the physical system against these distribution excursions.

---

## 6. Publication Figures and Visual Diagnostics

All four figures were generated by `src/decision_support/plotting.py` and saved to `reports/figures/`:

1. **`reports/figures/energy_dispatch_central.png`**:
   - Illustrates the hourly dispatch trajectory over 168 hours (7 days).
   - Shows stacked areas of direct renewable usage, battery discharging, and grid import exactly tracking total demand, alongside renewable curtailment and battery SOC dynamics.
2. **`reports/figures/energy_dispatch_uncertainty_bounds.png`**:
   - Displays the conformal dispatch envelopes across Conservative Stress, Central Expected, and Favorable Optimistic bounds.
   - Highlights the widening grid import band required to absorb demand volatility and renewable resource fluctuations.
3. **`reports/figures/capacity_optimization_surface.png`**:
   - Depicts the Pareto trade-off surface between total annualized cost and grid energy reliance across all 120 candidate configurations.
   - Identifies the 100% grid baseline, the Cost-Optimal mix, and the Autonomy-Optimal frontier.
4. **`reports/figures/scenario_comparison_metrics.png`**:
   - Grouped bar chart comparing total demand, available renewables, grid imports, and curtailment across the three uncertainty scenarios.

---

## 7. Limitations Preventing Real-World Site-Specific Claims

To uphold absolute scientific integrity, the following limitations are explicitly documented:

1. **Non-Co-Located Datasets**: The four underlying datasets describe four distinct geographic locations:
   - Solar Nowcasting: SKIPP'D (Stanford, California, USA)
   - Solar Irradiance: NASA POWER (Mumbai, Maharashtra, India; $19.076^\circ\text{N}, 72.878^\circ\text{E}$)
   - Wind Speed: NREL India Wind Toolkit (Gujarat, India; $22.091^\circ\text{N}, 70.334^\circ\text{E}$)
   - Electricity Demand: BDG2 `Panther_education_Rosalie` (Orlando, Florida, USA)
   The multimodal optimization in Stage 5 is a **proof-of-concept synthetic illustrative benchmark** demonstrating mathematical decision-support feasibility. It **does not represent any single physical microgrid installation**.
2. **Illustrative Power Proxies**: Wind power was computed via a generic 2 MW turbine power curve without wake losses, topography derating, or turbulence modeling. Solar conversion utilized an STC derate model without cell temperature dynamics or angle-of-incidence losses.
3. **Illustrative Cost Parameters**: Capital and grid tariffs are benchmark engineering proxies and must be replaced by site-specific contractor bids, balance-of-plant costs, and utility interconnection tariffs in commercial deployment.

---

## 8. Summary of Created and Modified Files

- **Source Code**:
  - `src/decision_support/__init__.py`: Public API exports.
  - `src/decision_support/energy_balance.py`: Energy balance simulation, battery storage dynamics, and strict conservation checks.
  - `src/decision_support/scenarios.py`: Provenance metadata schemas, STC solar conversion, and turbine power curve proxy.
  - `src/decision_support/optimizer.py`: Deterministic grid-search optimizer, multi-objective formulation, and CostModel with horizon amortization.
  - `src/decision_support/uncertainty_eval.py`: Conformal uncertainty sensitivity analysis and joint coverage disclosures.
  - `src/decision_support/plotting.py`: Matplotlib visualization routines.
- **Orchestration Script**:
  - `scripts/run_decision_support.py`: End-to-end execution, table export, figure generation, and registry logging.
- **Configuration**:
  - `configs/decision_support.yaml`: Simulation settings, grid limits, battery specifications, cost proxies, and search grid.
- **Tests**:
  - `tests/test_decision_support.py`: 15 comprehensive unit tests (conservation, boundaries, optimizer, validation, provenance).
- **Generated Artifacts**:
  - `reports/tables/energy_feasibility_scenarios.csv`: 120-candidate scenario optimization table.
  - `reports/tables/decision_support_summary.json`: Machine-readable summary metrics.
  - `reports/figures/energy_dispatch_central.png`
  - `reports/figures/energy_dispatch_uncertainty_bounds.png`
  - `reports/figures/capacity_optimization_surface.png`
  - `reports/figures/scenario_comparison_metrics.png`
  - `experiments/experiment_registry.csv`: Added rows 50, 51, 52.
