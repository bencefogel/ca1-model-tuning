import os
import pandas as pd
import numpy as np
from neuron import h


def measure_AMPA_current(model):
    """
    Measures AMPA receptor-mediated synaptic currents.

    Parameters:
        model (object): The NEURON model containing a list of AMPA synapses (`AMPAlist`).

    Returns:
        tuple:
            - AMPA (list): A list of `h.Vector` objects recording AMPA currents.
            - AMPA_segments (list): A list of segments where AMPA currents are recorded.
    """
    AMPA = []
    AMPA_segments = []

    for syn in model.AMPAlist:
        vec = h.Vector().record(syn._ref_i)
        AMPA.append(vec)
        AMPA_segments.append(syn.get_segment())
    return AMPA, AMPA_segments


def measure_NMDA_current(model):
    NMDA = []
    NMDA_segments = []

    for syn in model.NMDAlist:
        vec = h.Vector().record(syn._ref_i)
        NMDA.append(vec)
        NMDA_segments.append(syn.get_segment())
    return NMDA, NMDA_segments


def measure_GABA_current(model):
    GABA = []
    GABA_segments = []

    for syn in model.GABAlist:
        vec = h.Vector().record(syn._ref_i)
        GABA.append(vec)
        GABA_segments.append(syn.get_segment())
    return GABA, GABA_segments


def measure_GABA_B_current(model):
    GABA_B = []
    GABA_B_segments = []

    for syn in model.GABA_Blist:
        vec = h.Vector().record(syn._ref_i)
        GABA_B.append(vec)
        GABA_B_segments.append(syn.get_segment())
    return GABA_B, GABA_B_segments


def record_synaptic_currents(model):
    """
    Records synaptic currents for all synapse types (AMPA, NMDA, GABA, GABA-B).

    Parameters:
        model (object): The NEURON model containing lists of synapses (`AMPAlist`, `NMDAlist`, `GABAlist`, `GABA_Blist`).

    Returns:
        tuple:
            - synaptic_segments (dict): A dictionary where keys are synapse types and values are lists of segments.
            - synaptic_currents (dict): A dictionary where keys are synapse types and values are lists of `h.Vector` objects.
    """
    AMPA, AMPA_segments = measure_AMPA_current(model)
    NMDA, NMDA_segments = measure_NMDA_current(model)
    GABA, GABA_segments = measure_GABA_current(model)
    GABA_B, GABA_B_segments = measure_GABA_B_current(model)

    synaptic_currents = {
        'AMPA': AMPA,
        'NMDA': NMDA,
        'GABA': GABA,
        'GABA_B': GABA_B
    }

    synaptic_segments = {
        'AMPA': AMPA_segments,
        'NMDA': NMDA_segments,
        'GABA': GABA_segments,
        'GABA_B': GABA_B_segments
    }

    return synaptic_segments, synaptic_currents


def preprocess_synaptic_data(synaptic_segments, synaptic_currents, taxis_unique, index_unique):
    """
    Saves synaptic current data and segment information to disk.

    Parameters:
        synaptic_segments (dict): Keys are synapse types, and values are lists of segments where the synapse currents are recorded.
        synaptic_currents (dict): Keys are synapse types, and values are lists of recorded `h.Vector` objects.
        output_dir (str): Path to the directory where data will be saved.

    Outputs:
        - Segment information is saved as `.npy` files in the `synaptic_segments` subdirectory.
        - Current data is saved as `.npy` files in the `synaptic_currents` subdirectory.
    """
    segment_dict = {}
    current_dict = {}
    for synapse_type in synaptic_segments.keys():
        segments_array = np.array(synaptic_segments[synapse_type]).astype('str')

        # preprocess currents data (select unique indices, downsample to 5kHz)
        currents_array = np.array(synaptic_currents[synapse_type])
        currents_unique = currents_array[:, index_unique]

        x = int((max(taxis_unique)) * 5)  # Number of downsampled points for 5kHz downsampling, (max/1000)*5000
        taxis_downsampled = np.linspace(min(taxis_unique), max(taxis_unique), x)
        currents_downsampled = np.array([
            np.interp(taxis_downsampled, taxis_unique, row) for row in currents_unique
        ])
        segment_dict[synapse_type] = segments_array
        current_dict[synapse_type] = currents_downsampled
    return segment_dict, current_dict
