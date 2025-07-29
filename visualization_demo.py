import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def get_vm(data: dict, segment: str) -> np.array:
    segment_idx = np.where(data['membrane_potential_data'][0] == segment)[0]
    return data['membrane_potential_data'][1][segment_idx, :].flatten()

amp = 0.5
gcar = 0.006
gkslow = 0.001
output_dir = "L:/model_optimization/current_injection"

fname = f'{amp}pA_iinj_gcar{gcar}_gkslow{gkslow}.pkl'
filepath = os.path.join(output_dir, fname)

with open(filepath, 'rb') as f:
    simulation_data = pickle.load(f)

t = simulation_data['taxis']
vm_soma = get_vm(simulation_data, 'soma(0.5)')
vm_dend = get_vm(simulation_data, 'dend5_01111111111111111(0.5)')
plt.plot(t, vm_soma, color='black', label='Soma Vm')
plt.plot(t, vm_dend, color = 'maroon', label='Dendrite Vm')
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.legend()
plt.show()