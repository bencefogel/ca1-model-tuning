import numpy as np
from neuron import h
import os


def record_membrane_potential():
    """
    Records the membrane potential from all segments in all sections of the NEURON model.

    Returns:
        tuple:
            - v_segments (list): A list of segment objects where the membrane potential was recorded.
            - v (list): A list of `h.Vector` objects containing the recorded membrane potential data.
    """
    v = []
    v_segments = []

    for sec in h.allsec():
        for seg in sec.allseg():
            v_segments.append(seg)
            v.append(h.Vector().record(seg._ref_v))
    return v_segments, v


def preprocess_membrane_potential_data(v_segments, v, taxis_unique, index_unique):
    """
    Saves recorded membrane potential data and corresponding segment information to disk.

    Parameters:
        v_segments (list): A list of segment objects where the membrane potential was recorded.
        v (list): A list of `h.Vector` objects containing the recorded membrane potential data.
        output_dir (str): The path to the directory where the data will be saved.

    Outputs:
        - Segment information is saved as `segments.npy` in the `membrane_potential_data` subdirectory.
        - Membrane potential data is saved as `v.npy` in the same subdirectory.
    """
    segments_array = np.array([str(seg) for seg in v_segments])

    # preprocessing (select unique indices, downsample to 5kHz)
    potential_array = np.array(v)
    potential_array_unique = potential_array[:, index_unique]  # select unique indices

    x = int((max(taxis_unique)) * 5)  # Number of downsampled points for 5kHz downsampling, (max/1000)*5000
    taxis_downsampled = np.linspace(min(taxis_unique), max(taxis_unique), x)
    potential_downsampled = np.array([
        np.interp(taxis_downsampled, taxis_unique, row) for row in potential_array_unique
    ])
    return segments_array, potential_downsampled

