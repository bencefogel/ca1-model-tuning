# Model Tuning with Optuna (trunk vs. non-trunk dendritic Ca2+-spikes)

This repository performs automated parameter optimization to maximize calcium Ca2+ influx in the trunk of a CA1 pyramidal neuron model while minimizing calcium influx in oblique dendrites, using the Optuna framework for hyperparameter tuning.

---

## Overview

The pipeline simulates dendritic currents using a custom `CurrentscapePipeline`, then evaluates performance via:

- `trunk_score()`: quantifies Ca2+ current in the distal trunk
- `oblique_penalty()`: penalizes Ca2+ current in oblique branches

The goal is to **maximize trunk Ca2+ influx** while **suppressing unwanted oblique branch activation**.

---

## Files

- `main_full_pipeline.py`: orchestrates the optimization loop
- `CurrentscapePipeline.py`: defines neuron model and simulation logic
- `custom_objective_scores.py`: defines custom scoring functions

---

## How to Run the Pipeline
### 1. **Install Requirements**

Use Python 3.9+ and install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 2. **Compile Mechanism Files**

Before running simulations, you need to compile NEURON's `.mod` files.
Navigate to the folder containing the `.mod` files:

```bash
cd simulator/model/density_mechs
```
And run the `nrnivmodl` tool from NEURON.

If NEURON has trouble finding compiled mechanisms, consider copying them into the root folder and re-running `nrnivmodl`.

More information about NEURON: https://neuron.yale.edu/neuron<br>
More information about working with .mod files: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3263<br>
More information about compiling .mod files: https://nrn.readthedocs.io/en/latest/guide/faq.html#how-do-i-compile-mod-files

---

### 3. Run Parameter search

1. **Suggest Parameters**: 
    - `gcar`, `gkslow`, `gcar_trunk`, and `gkslow_trunk` are optimized.
2. **Run Simulation**: 
    - A current injection protocol is simulated via `CurrentscapePipeline`.
3. **Score Simulation**:
    - High `trunk_score` → rewarded
    - High `oblique_penalty` → penalized
4. **Optimize**:
    - Optuna runs 50 trials to maximize the objective.
5. **Visualization**:
    - Configure inputs and run `visualization_optuna.py` to visualize the top 5 scoring parameter sets.