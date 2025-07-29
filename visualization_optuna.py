import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from CurrentscapePipeline import CurrentscapePipeline
import optuna

# === SETTINGS ===
output_dir = "L:/model_optimization/optuna_top_trials_duration_penalty"
os.makedirs(output_dir, exist_ok=True)

study_name = "trunk_ca_opt_duration_penalty"
storage = "sqlite:///trunk_ca_opt_duration_penalty.db"
TOP_N = 5
amp = 1
delay = 30
dur = 100
tstop = 150

TRUNK_SEG = "dend5_01111111111111111"
OBLIQUE_SEG = "dend5_01111111100"

def get_spike_intervals(icar, taxis, delay, dur, threshold_ratio=0.5):
    t_baseline = taxis < delay
    t_stim_mask = (taxis >= delay) & (taxis <= delay + dur)

    icar_baseline = np.mean(icar[t_baseline])
    icar_peak = np.min(icar[t_stim_mask])

    threshold = icar_baseline + threshold_ratio * (icar_peak - icar_baseline)

    t_stim = taxis[t_stim_mask]
    icar_stim = icar[t_stim_mask]

    dt = np.mean(np.diff(t_stim))
    below_thresh = icar_stim < threshold

    starts = np.where(np.diff(below_thresh.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(below_thresh.astype(int)) == -1)[0] + 1

    if below_thresh[0]:
        starts = np.insert(starts, 0, 0)
    if below_thresh[-1]:
        ends = np.append(ends, len(below_thresh))

    durations = (ends - starts) * dt
    return starts, ends, durations, t_stim

def get_vm(data: dict, segment: str) -> np.ndarray:
    idx = np.where(data['membrane_potential_data'][0] == segment)[0]
    return data['membrane_potential_data'][1][idx, :].flatten()

def get_icar(data: dict, segment: str) -> np.ndarray:
    idx = np.where(data['intrinsic_data'][0]['car'] == f'{segment}(0.5)')[0]
    return np.array(data['intrinsic_data'][1]['car'][idx, :]).flatten()

def run_sim(params: dict):
    pipeline = CurrentscapePipeline.current_injection(
        gcar=params["gcar"],
        gkslow=params["gkslow"],
        gcar_trunk=params["gcar_trunk"],
        gkslow_trunk=params["gkslow_trunk"],
        target=TRUNK_SEG,
        amp=amp,
        dur=dur,
        delay=delay,
        tstop=tstop
    )
    return pipeline.run_current_injection()

# === LOAD STUDY ===
study = optuna.load_study(study_name=study_name, storage=storage)
top_trials = sorted(
    [t for t in study.trials if t.value is not None],
    key=lambda t: t.value,
    reverse=True
)[:TOP_N]

for i, trial in enumerate(top_trials):
    print(f"\n=== Trial {i+1} | Score: {trial.value:.3f} ===")
    params = trial.params
    for k, v in params.items():
        print(f"{k}: {v:.5f}")

    try:
        data = run_sim(params)
        t = data['taxis']
        vm_soma = get_vm(data, "soma(0.5)")
        vm_trunk = get_vm(data, f"{TRUNK_SEG}(0.5)")
        icar_trunk = get_icar(data, TRUNK_SEG)
        icar_oblique = get_icar(data, OBLIQUE_SEG)

        # Detect Ca spike properties
        starts, ends, durations, t_stim = get_spike_intervals(icar_trunk, t, delay, dur)

        # === PLOT ===
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

        axs[0, 0].plot(t, vm_soma, label="Soma Vm", color="black")
        axs[0, 1].plot(t, vm_trunk, label="Trunk Vm", color="darkred")
        axs[1, 0].plot(t, icar_trunk, label="Ca Trunk", color="blue")
        axs[1, 1].plot(t, icar_oblique, label="Ca Oblique", color="teal")

        for ax in axs.flat:
            ax.axvspan(delay, delay+dur, color="gray", alpha=0.2)
            ax.legend()
            ax.set_xlabel("Time (ms)")

        axs[0, 0].set_ylabel("Vm (mV)")
        axs[1, 0].set_ylabel("ICa (nA)")
        fig.suptitle(
            f"Trial {i+1} | Score={trial.value:.3f} | gcar={params['gcar']:.4f}, gcar_trunk={params['gcar_trunk']:.4f}",
            fontsize=12
        )

        # Plot shading for longest duration spike
        if len(durations) > 0:
            idx_longest = np.argmax(durations)
            d_start = t_stim[starts[idx_longest]]
            d_end = t_stim[ends[idx_longest] - 1]
            axs[1, 0].axvspan(d_start, d_end, color='purple', alpha=0.2, label='Longest Ca spike duration')
            axs[1, 0].legend()

        # Get y limits for first row and set both axes to their combined range
        ymin_vm = min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0])
        ymax_vm = max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1])
        axs[0, 0].set_ylim(ymin_vm, ymax_vm)
        axs[0, 1].set_ylim(ymin_vm, ymax_vm)

        # Same for second row (ICa)
        ymin_ica = min(axs[1, 0].get_ylim()[0], axs[1, 1].get_ylim()[0])
        ymax_ica = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
        axs[1, 0].set_ylim(ymin_ica, ymax_ica)
        axs[1, 1].set_ylim(ymin_ica, ymax_ica)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        outname = f"trial_{i+1}_Vm_ICa.png"
        plt.savefig(os.path.join(output_dir, outname))
        plt.close()

        # Optional: Save simulation
        pkl_name = f"trial_{i+1}_simdata.pkl"
        with open(os.path.join(output_dir, pkl_name), "wb") as f:
            pickle.dump(data, f)

    except Exception as e:
        print(f"Failed to run trial {i+1}: {e}")
