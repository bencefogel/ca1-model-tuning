import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

output_dir = "L:/model_optimization/current_injection/trunk_increasing2x"
cc_list = np.linspace(1, 0, num=11)
amp_list = np.arange(0.2, 1.01, 0.2)

def get_vm(data: dict, segment: str) -> np.array:
    segment_idx = np.where(data['membrane_potential_data'][0] == segment)[0]
    return data['membrane_potential_data'][1][segment_idx, :].flatten()


for i, cc in enumerate(cc_list):
    gcar = np.round(0.006 * cc, 4)
    gkslow = np.round(0.001 * cc, 4)
    gcar_trunk = np.round(0.006 + ((1 - cc) * 0.006), 4)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axs[0].set_title(f'Soma Vm')
    axs[1].set_title(f'Dendrite Vm')

    for amp in amp_list:
        x_shift_step = 25  # in ms
        y_shift_step = 25  # in mV
        fname = f'{np.round(amp,1)}pA_iinj_gcar{gcar}_trunkgcar{gcar_trunk}_gkslow{gkslow}.pkl'
        filepath = os.path.join(output_dir, fname)

        if not os.path.exists(filepath):
            print(f"Missing file: {fname}")
            continue

        with open(filepath, 'rb') as f:
            simulation_data = pickle.load(f)

        t = simulation_data['taxis']
        v_soma = get_vm(simulation_data, 'soma(0.5)')
        v_dend = get_vm(simulation_data, 'dend5_01111111111111111(0.5)')

        color = plt.cm.viridis((amp - min(amp_list)) / (max(amp_list) - min(amp_list)))

        x_shift = amp_list.tolist().index(amp) * x_shift_step
        y_shift = amp_list.tolist().index(amp) * y_shift_step

        axs[0].plot(t + x_shift, v_soma + y_shift, label=f'{np.round(amp, 1)}nA', color=color)
        axs[1].plot(t + x_shift, v_dend + y_shift, label=f'{np.round(amp, 1)}nA', color=color)

    axs[0].set_xlabel('Time (ms)')
    axs[1].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Vm (mV)')

    # Remove grid
    for ax in axs:
        ax.grid(False)

    # Remove top and right spines for soma
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # Remove top, right, and left spines for dendrite
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].tick_params(left=False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)

    fig.suptitle(f'Vm response (250 ms iinj)\n gcar={gcar}, trunk_gcar={gcar_trunk} gkslow={gkslow}', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])

    outpath = os.path.join(output_dir, f'Vm_gcar{gcar}_trunkgcar{gcar_trunk}_gkslow{gkslow}.png')
    plt.savefig(outpath)
    plt.close()
