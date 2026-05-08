# LLTEM SINDy Conflict Resolution Analysis

Sparse Identification of Nonlinear Dynamics (SINDy) applied to the MIT Lincoln Laboratory Terminal Encounter Model V1.0 dataset, targeting physics-grounded equation discovery for UAM conflict resolution near Class D airports.

---

## Research Objective

Discover interpretable governing equations for aircraft conflict dynamics in terminal airspace using the SINDy framework. The identified equations are intended to characterize the kinematics of well-clear violations (DO-365C) and inform conflict resolution logic for unmanned ownships on straight-in IFR approaches.

---

## Dataset

**LLTEM V1.0** — MIT Lincoln Laboratory Terminal Encounter Model  
- 1,000,000 simulated pairwise terminal airspace encounters  
- Ownship: unmanned aircraft on straight-in IFR approach  
- Intruder: manned aircraft landing (intent=1) or taking off (intent=2) at Class D airport  
- Raw trajectories: 15-column CSV per aircraft at 1 Hz  
- Metadata: `terminal_encounter_info_20200630.csv` (HMD, VMD, tCPA, speeds, altitudes, track angles)

**Well-clear thresholds (DO-365A / ACAS Xu):**

| Dimension | Threshold |
|-----------|-----------|
| Horizontal Miss Distance (HMD) | < 4,000 ft |
| Vertical Miss Distance (VMD) | < 450 ft |

Encounters violating both simultaneously are classified as **conflict scenarios** (~19.7% of the full dataset).

---

## Notebook Structure — `LLTEM_analysis 2.ipynb`

### Phase 1 — Setup & Exploration (Cells 1–2)
- Imports, ZIP loader (`load_encounter_pair`), 5-encounter geometry spot-check

---

### Phase 2 — Section 4.2: Data Integrity & Preprocessing

#### 4.2.1 — Dataset Validation (4 steps)
| Step | Content |
|------|---------|
| 1 | File-count integrity check (1M ownship + 1M intruder CSVs) |
| 2 | Metadata distributions: HMD, VMD, tCPA, altitude, speed, intruder intent |
| 3 | Conflict filter: HMD < 4,000 ft AND \|VMD\| < 450 ft → **~197,000 conflicts** |
| 4 | Simultaneous violation breakdown (both / HMD-only / VMD-only / neither) |

#### 4.2.2 — SINDy Preprocessing Pipeline (5 steps)
| Step | Content |
|------|---------|
| 1 | Geometry classification (head-on \|Δψ\| > 135°, overtaking < 45°, crossing otherwise); stratified 6,000-encounter sample; 5,000 train / 1,000 val split |
| 2 | State extraction: 26-variable vector per timestep (positions, velocities, heading, range, bearing, closure rate) |
| 3 | Temporal alignment: ±60 s window around CPA, 1 Hz uniform resampling (121 timesteps) |
| 4 | Derivative estimation: Savitzky-Golay filter (window=7, order=3) for simultaneous smoothing and dẋ/dt |
| 5 | Z-score normalization (fit on training set only); HDF5 export; derivative reconstruction quality check |

**26-variable state vector (`SINDY_VARS`):**
```
own_x  own_y  own_z  own_vx  own_vy  own_vz  own_psi  own_gamma
int_x  int_y  int_z  int_vx  int_vy  int_vz  int_psi  int_gamma
rel_x  rel_y  rel_z  rel_vx  rel_vy  rel_vz
range_h  range_3d  bearing  closure_rate
```

**Pipeline result:** 4,247 training trajectories, 826 validation trajectories (927 discarded — CPA too close to trajectory edge).

---

### Phase 2 — Section 4.3: EDA & Feature Engineering

#### 4.3.1 — Encounter Characterization (4 steps)
| Step | Content |
|------|---------|
| 1 | Geometry distribution: head-on 56.5%, overtaking 28.9%, crossing 14.6%; HMD/VMD by geometry |
| 2 | State variable statistics at CPA (mean/std/min/max, all 26 vars) |
| 3 | Derivative activity ranking: mean \|ẋ\| across ±60 s window — rel_x, int_x, range_h dominate |
| 4 | Feature importance: Pearson r with min 3D conflict range — range_h/range_3d top predictors |

#### 4.3.2 — Feature Engineering for SINDy (3 steps)

**Reduced 2D horizontal state (`VARS_2D`, 11 variables):**
```
rel_x  rel_y  own_vx  own_vy  own_psi
int_vx  int_vy  int_psi  range_h  bearing  closure_rate
```

**Custom library — 84 terms total:**

| Group | Terms | Justification |
|-------|-------|---------------|
| Constant | 1 | Mean drift absorption |
| Linear | 11 | Exact kinematics: d(rel_x)/dt = int_vx − own_vx, d(range_h)/dt = −closure_rate |
| Quadratic | 66 | Closure-rate acceleration, velocity products, geometry×heading coupling |
| Trig (angular vars only) | 6 | NED decomposition: own_vx = v·cos(ψ), own_vy = v·sin(ψ); bearing-rate equation |

**Excluded:** 1/range_h (singular at CPA), degree-3 terms (unphysical for constant-speed segments; library grows to 370+ terms)

| Step | Content |
|------|---------|
| Library design | NED convention derivation, physics motivation per group, hand-built `LIBRARY_FUNCS`, term catalogue |
| PCA | 513,887 samples × 26 vars; 95% variance in 3 components (rel_x/int_x dominate PC1, range in PC2) |
| Validation | Θ(X) condition number κ = 4.88e+22 on raw state → confirmed normalization required |

---

### Phase 3 — Section 4.4: SINDy Model Development

#### 4.4.1 — Baseline SINDy Implementation (4 steps)

| Step | Content |
|------|---------|
| 1 | `GeneralizedLibrary([PolynomialLibrary(degree=2), FourierLibrary(n_frequencies=1)], inputs_per_library=[all_11, [4,7,9]])` — trig restricted to own_psi, int_psi, bearing only |
| 2 | λ cross-validation: STLSQ at λ ∈ {0.005, 0.01, 0.05, 0.1} on 500-encounter subset; select by val-set reconstruction MSE |
| 3 | Full fit: STLSQ + SR3 on all 4,247 training trajectories; equation display; 11×84 coefficient heatmap |
| 4 | Evaluation: per-variable MSE/R², physical plausibility check (d(own_psi)/dt ≈ 0), prediction horizon via `model.simulate()` |

**Key design choice — `inputs_per_library`:** restricts `FourierLibrary` to angular variable indices `[4, 7, 9]`, preventing unphysical terms like sin(rel_x) or sin(closure_rate) that would be generated by a naive `GeneralizedLibrary` construction.

**Optimizer:** STLSQ (Sequential Threshold Least Squares) as primary; SR3 as backup for high-κ library matrices. Both are version-safe — SR3 parameter names are detected via `inspect.signature` at runtime.

---

## Planned Work

### 4.4.2 — Geometry-Stratified Fitting
- Fit separate SINDy models for head-on, overtaking, crossing
- Compare coefficient matrices: identify shared vs geometry-specific structure
- Visualize sparsity pattern differences across geometries

### 4.4.3 — Model Validation & Stability
- Prediction horizon benchmarking (simulate from t=−60 s, measure RMSE growth)
- Bootstrap coefficient stability over random train subsets
- Residual analysis: check for systematic structure in SINDy fit residuals

### 4.5 — Phase 4: Conflict Resolution Application
- Use discovered equations to define conflict severity metrics
- Test equation-derived maneuver recommendations against LLTEM scenarios
- Compare SINDy-derived decision boundaries with DO-365A thresholds

### 4.6 — Phase 5: Robustness & Generalization
- Cross-validation across geometry types
- Sensitivity to SG filter parameters (window, order)
- Ablation: full 26-var state vs reduced 11-var 2D state

---

## Dependencies

```
pandas
numpy
matplotlib
scipy          # savgol_filter, interp1d
scikit-learn   # PCA
h5py           # HDF5 dataset export
pysindy >= 1.3 # GeneralizedLibrary with inputs_per_library
```

Install:
```bash
pip install pandas numpy matplotlib scipy scikit-learn h5py pysindy
```

---

## Data Files Required

| File | Description |
|------|-------------|
| `terminal_encounter_state_data_20200630.zip` | 2M trajectory CSVs (ownship + intruder) |
| `terminal_encounter_info_20200630.csv` | 1M-row metadata (HMD, VMD, tCPA, speeds, geometry) |
| `lltem_sindy_dataset.h5` | Generated by Section 4.2.2 — normalized train/val arrays + norm params |

---

## Run Order

Execute cells top to bottom. The pipeline cell in **4.2.2** (~40 s for 6,000 encounters) must complete before any **4.3.x** or **4.4.x** cells produce meaningful output. Subsequent sections guard on `if not train_data` and print a reminder if the pipeline has not been run.
