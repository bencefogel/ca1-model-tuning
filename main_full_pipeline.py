import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
import optuna
import json
import optuna.visualization as vis

from tqdm import tqdm
from CurrentscapePipeline import CurrentscapePipeline
from custom_objective_scores import trunk_score, oblique_penalty

def get_icar(data: dict, segment: str) -> np.ndarray:
    """Extract Ca2+ current trace from a given segment."""
    segment_idx = np.where(data['intrinsic_data'][0]['car'] == f'{segment}(0.5)')[0]
    return np.array(data['intrinsic_data'][1]['car'][segment_idx, :]).flatten()


def objective(trial):
    # Suggested parameter ranges
    gcar = trial.suggest_float("gcar", 0.0, 0.015)
    gkslow = trial.suggest_float("gkslow", 0.0, 0.01)
    gcar_trunk = trial.suggest_float("gcar_trunk", 0.0, 0.015)
    gkslow_trunk = trial.suggest_float("gkslow_trunk", 0.0, 0.01)

    amp = 1  # 1000 pA
    delay = 30  # ms
    dur = 100  # ms
    tstop = 150  # ms

    pipeline = CurrentscapePipeline.current_injection(
        gcar=gcar,
        gkslow=gkslow,
        gcar_trunk=gcar_trunk,
        gkslow_trunk=gkslow_trunk,
        target="dend5_01111111111111111",  # distal trunk
        amp=amp,
        dur=dur,
        delay=delay,
        tstop=tstop
    )

    try:
        simulation_data = pipeline.run_current_injection()
        taxis = simulation_data['taxis']

        # Get Ca currents
        icar_trunk = get_icar(simulation_data, 'dend5_01111111111111111')
        icar_oblique = get_icar(simulation_data, 'dend5_01111111100')

        trunk_score = trunk_score(icar_trunk, taxis, delay, dur)
        oblique_score = oblique_penalty(icar_oblique, taxis, delay, dur)

        # Combine: promote trunk Ca spike, penalize oblique Ca influx
        f = 1 # weighting factor for oblique penalty
        total_score = trunk_score - f * oblique_score
        print(f"Trial: Trunk={trunk_score:.3f}, Oblique={oblique_score:.3f}, Total={total_score:.3f}")
        return total_score

    except Exception as e:
        print(f"Simulation failed: {e}")
        return -1e9

study_name = "trunk_ca_opt_duration_penalty"
storage = "sqlite:///trunk_ca_opt_duration_penalty.db"

study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

for _ in tqdm(range(50), desc="Optimizing"):
    study.optimize(objective, n_trials=1, catch=(Exception,))


# Show optimization progress
fig1 = vis.plot_optimization_history(study)
fig1.show()

# Show parameter importance
fig2 = vis.plot_param_importances(study)
fig2.show()
