import numpy as np
import altair as alt
from currentscape_visualization.utils import *


def plot_currentscape(part_pos: pd.DataFrame, part_neg: pd.DataFrame, vm: np.array, taxis: np.array, tmin: int, tmax: int,
                      return_segs: bool=False, segments_preselected: bool=True,
                      vmin: int=-69, vmax: int=20, partitionby: str='type'):
    """
    Generates and saves a currentscape plot.

    Args:
    part_pos : pd.DataFrame
        The data frame representing positive currents.

    part_neg : pd.DataFrame
        The data frame representing negative currents.

    vm : np.array
        Array of membrane potential values.

    taxis : np.array
        Array of time axis values corresponding to the data.

    tmin : int
        Start timepoint of the visualization.

    tmax : int
        End timepoint of the simulation

    filename : str
        The name of the file where the currentscape will be saved.

    return_segs : bool, default=False
        If True, returns currentscape data instead of only saving the file.

    segments_preselected : bool, default=True
        If True, assumes the provided data contains already selected timepoints

    vmin : int, default=-69
        Minimum membrane potential value for visualization.

    vmax : int, default=-65
        Maximum membrane potential value for visualization.

    partitionby : str, default='type'
        Sets partitioning strategy. Can be 'type' or 'region'-specific.

    Returns
    list of pd.DataFrame, np.array or None
        A list containing the partitioned negative currents, positive currents, and membrane potential values,
        if return_segs is True. Otherwise, None.
    """
    print("Generating currentscape...")
    if (segments_preselected):
        part_pos_seg = part_pos
        part_neg_seg = part_neg
        t_seg = taxis
        vm_seg = vm
    else:
        segment_indexes = np.flatnonzero((taxis > tmin) & (taxis < tmax))
        part_pos_seg = part_pos[segment_indexes]
        part_neg_seg = part_neg[segment_indexes]
        t_seg = taxis[segment_indexes]
        vm_seg = vm

    # Create charts
    totalpos = create_currsum_pos_chart(part_pos_seg, t_seg)
    currshares_pos, currshares_neg = create_currshares_chart(part_pos_seg, part_neg_seg, t_seg, partitionby)
    vm_chart = create_vm_chart(vm_seg, t_seg, vmin, vmax)

    # Create currentscape
    if (return_segs):
        return [part_neg_seg, part_pos_seg, vm_seg]
    else:
        return combine_charts(vm_chart, totalpos, currshares_pos, currshares_neg)