import os
import pandas as pd
import numpy as np
from neuron import h

# Dictionary mapping current types to their corresponding NEURON attributes
current_types = {
        'nax': '_ref_ina_nax',
        'nad': '_ref_ina_nad',
        'car': '_ref_ica_car',
        'kdr': '_ref_ik_kdr',
        'kap': '_ref_ik_kap',
        'kad': '_ref_ik_kad',
        'kslow': '_ref_ik_kslow',
        'passive': '_ref_i_pas',
        'capacitive': '_ref_i_cap',
    }


def measure_intrinsic(seg, current_types):
    """
    Measures intrinsic currents in a given segment.

    Parameters:
        seg (object): A NEURON segment object to record from.
        current_types (dict): A dictionary mapping current type names to their NEURON reference attributes.

    Returns:
        dict: A dictionary where keys are current types and values are recorded `h.Vector` objects containing the data.
    """
    recorded_vectors = {}
    for current, ref_attr in current_types.items():
        if hasattr(seg, ref_attr):
            vec = h.Vector()
            vec.record(getattr(seg, ref_attr))
            recorded_vectors[current] = vec
    return recorded_vectors


def record_intrinsic_currents():
    """
    Records intrinsic currents for all segments in all sections of the NEURON model.
    Iterates through all segments in all sections and records intrinsic currents based on the defined current types.

    Returns:
        tuple:
            - intrinsic_segments (dict): Keys are current types, and values are lists of segments where the current type is present.
            - intrinsic_currents (dict): Keys are current types, and values are lists of `h.Vector` objects for the recorded data.
    """
    intrinsic_currents = {current: [] for current in current_types}
    intrinsic_segments = {current: [] for current in current_types}

    for sec in h.allsec():
        for seg in sec.allseg():
            recorded = measure_intrinsic(seg, current_types)
            for current, vec in recorded.items():
                intrinsic_currents[current].append(vec)
                intrinsic_segments[current].append(seg)
    return intrinsic_segments, intrinsic_currents


def preprocess_intrinsic_data(intrinsic_segments, intrinsic_currents, taxis_unique, index_unique):
    """
    Saves recorded intrinsic current data and segment information to disk.

    Parameters:
        intrinsic_segments (dict): Keys are current types, and values are lists of segments where the current type is present.
        intrinsic_currents (dict): Keys are current types, and values are lists of recorded `h.Vector` objects.
        output_dir (str): Path to the directory where data will be saved.

    Outputs:
        - Segment information is saved as `.npy` files in the `intrinsic_segments` subdirectory.
        - Current data is saved as `.npy` files in the `intrinsic_currents` subdirectory.
    """
    segment_dict = {}
    current_dict = {}
    for current_type in intrinsic_segments.keys():
        segments_array = np.array(intrinsic_segments[current_type]).astype('str')

        # preprocess currents data (select unique indices, downsample to 5kHz)
        currents_array = np.array(intrinsic_currents[current_type])
        currents_unique = currents_array[:,index_unique]

        x = int((max(taxis_unique)) * 5)  # Number of downsampled points for 5kHz downsampling, (max/1000)*5000
        taxis_downsampled = np.linspace(min(taxis_unique), max(taxis_unique), x)
        currents_downsampled = np.array([
            np.interp(taxis_downsampled, taxis_unique, row) for row in currents_unique
        ])
        segment_dict[current_type] = segments_array
        current_dict[current_type] = currents_downsampled
    return segment_dict, current_dict
