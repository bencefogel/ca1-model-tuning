from neuron import h, gui
import numpy as np

from simulator.model.utils.record_intrinsic import record_intrinsic_currents, preprocess_intrinsic_data
from simulator.model.utils.record_synaptic import record_synaptic_currents, preprocess_synaptic_data
from simulator.model.utils.record_membrane_potential import record_membrane_potential, preprocess_membrane_potential_data


def simulate(model, tstop):
    h.CVode().active(True)
    h.CVode().atol((1e-3))

    # Record time array
    trec = h.Vector()
    trec.record(h._ref_t)

    v_segments, v = record_membrane_potential()
    intrinsic_segments, intrinsic_currents = record_intrinsic_currents()
    # synaptic_segments, synaptic_currents = record_synaptic_currents(model)

    h.celsius = 35
    h.finitialize(-68.3)

    h.continuerun(tstop)

    taxis = np.array(trec)
    taxis_unique, index_unique = np.unique(taxis, return_index=True)
    x = int((max(taxis_unique)) * 5)
    taxis_downsampled = np.linspace(min(taxis_unique), max(taxis_unique), x)

    v_segments, v_arrays = preprocess_membrane_potential_data(v_segments, v, taxis_unique, index_unique)
    intrinsic_segments, intrinsic_arrays = preprocess_intrinsic_data(intrinsic_segments, intrinsic_currents, taxis_unique, index_unique)
    # synaptic_segments, synaptic_arrays = preprocess_synaptic_data(synaptic_segments, synaptic_currents, taxis_unique, index_unique)

    simulation_data = {'membrane_potential_data': [v_segments, v_arrays],
                       'intrinsic_data': [intrinsic_segments, intrinsic_arrays],
                       'taxis': taxis_downsampled}
    return simulation_data
