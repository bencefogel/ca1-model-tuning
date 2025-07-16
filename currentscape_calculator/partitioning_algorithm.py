import os
import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from currentscape_calculator.partitioning_order import create_directed_graph, get_partitioning_order



def partition_iax(im: pd.DataFrame, iax: pd.DataFrame, timepoints: list, target: str, partition_by: str,
                  regions_list_directory: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the axial currents based on the target node and partitioning criteria.
    It prepares region-specific indices and recalculates membrane if required.

    Args:
        im : DataFrame
            A DataFrame containing membrane currents indexed by segments and current type.
        iax : DataFrame
            A DataFrame containing axial currents indexed by reference and parent segments.
        timepoints : list
            A list of time points at which the partitioning is performed.
        target : str
            The name of the target node segment for partitioning.
        partition_by : str
            Partitioning strategy. Can be either 'type' or 'region'
        regions_list_directory : str
            Directory path containing data about regions for each dendritic branch. This is necessary when partitioning
            by 'region'.

    Returns
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames:
            1. Positive membrane currents indexed by the target node and specified timepoints.
            2. Negative membrane currents indexed by the target node and specified timepoints.
    """

    if (target != 'soma'):
        print('updating current files to the new target node:', target)
        ## we need to start with modifying the DataFrames containing the axial and membrane currents
        ## first: membrane currents
        ##        merging the segments belonging to the target section
        im = merge_dendritic_section_imembrane(im, target)
        ## second: axial currents
        ##         removing internal nodes of the target
        iax = merge_dendritic_section_iax(iax, target)
        ##         changing the axial current directions between the soma and the target to reflect the new target node
        iax = update_root_node(iax, target)
        print('current files updated')

    if (partition_by == 'region'):
        if (regions_list_directory is None):
            raise ValueError(
                'The directory containing the files defining the region for each dendritic branch is missing.')

        ## reindexing the currents by their region... / loosing their identity
        df_index_orig = pd.DataFrame((np.array((im.index.get_level_values(0), im.index.get_level_values(1))).T),
                                     columns=['segment', 'itype'])
        df_index_region_specific = create_region_specific_index(df_index_orig, regions_list_directory)
        multiindex = pd.MultiIndex.from_frame(df_index_region_specific)
        im = pd.DataFrame(data=im.values, index=multiindex)

    # Separate DataFrames for positive and negative membrane currents
    im_pos = im.clip(lower=0)  # Positive currents only
    im_neg = im.clip(upper=0)  # Negative currents only
    if (partition_by == 'region'):
        print('Recalculating membrane currents by region...')
        im_pos = calc_im_by_region(im_pos)
        im_neg = calc_im_by_region(im_neg)

    for tp in tqdm(timepoints):
        dg = create_directed_graph(iax, tp)

        partitioning_order_out = get_partitioning_order(dg, target,
                                                        'out')  # axial current always POSITIVE, we use the POSITIVE membrane currents
        for segment_pair in partitioning_order_out:
            ref = segment_pair[0]
            par = segment_pair[1]
            iax_tp = iax.loc[(ref, par), tp]
            im_tp = im_pos.loc[ref, tp]
            sum_im_tp = im_tp.sum()
            # iax_tp either POSITIVE or NEGATIVE
            if ((iax_tp > 0) & (sum_im_tp != 0)):
                partition_iax_single(ref, par, tp, im_pos, iax_tp)

        partitioning_order_in = get_partitioning_order(dg, target,
                                                       'in')  # axial current always NEGATIVE, we use the NEGATIVE membrane currents
        for segment_pair in partitioning_order_in:
            ref = segment_pair[0]
            par = segment_pair[1]
            iax_tp = iax.loc[(ref, par), tp]
            im_tp = im_neg.loc[ref, tp]
            sum_im_tp = im_tp.sum()
            # iax_tp either POSITIVE or NEGATIVE
            if ((iax_tp < 0) & (sum_im_tp != 0)):
                partition_iax_single(ref, par, tp, im_neg, iax_tp)

    return im_pos.iloc[:, timepoints].loc[target], im_neg.iloc[:, timepoints].loc[target]


def merge_dendritic_section_imembrane(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    Merges data for a specific dendritic section, summing  the values for each `itype` across the segments of the section

    Parameters:
    ----------
    df : pd.DataFrame
        A pandas DataFrame where rows are indexed by a multi-level index. The first level of the index is a segment
        identifier, and the second level represents `itype`.

    section : str
        The name of the dendritic section to be processed. This will be used to select the rows that belong to the
        given section.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame that combines the original data excluding the selected dendritic section and the summed data
        for that section grouped by `itype`. The new DataFrame has the dendritic segment and `itype` as a two-level index.
    """

    df_dend = df[df.index.get_level_values(0).str.startswith(f'{section}(')]  # select all rows belonging to the given segment
    df_summed_by_itype = df_dend.groupby(level='itype').sum()  # sum dataframe by current type for each time point
    df_summed_by_itype = df_summed_by_itype.reset_index()
    df_summed_by_itype['segment'] = section
    df_summed_by_itype = df_summed_by_itype.set_index(['segment', 'itype'])

    # Update original dataframe with the merged dendritic segment
    df_merged_dendritic_segment = pd.concat([df.drop(df_dend.index), df_summed_by_itype], axis=0)
    return df_merged_dendritic_segment


def merge_dendritic_section_iax(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
   This function selects the axial current connections that are external to the specified dendritic section
   (i.e., connections between parent and children nodes) and merges them back into the dataframe after
   renaming and removing internal connections.

   Args:
       df : pd.DataFrame
           The dataframe containing axial current connections, with a multi-level index that includes 'ref'
           (reference) and 'par' (parent) segments.
       section : str
           The dendritic section identifier for which external axial current connections are to be merged.

   Returns:
       pd.DataFrame
           A dataframe with the external axial current connections of the specified dendritic segment merged back
           into the original dataframe, while internal connections are removed.

   Notes:
       - Internal axial current connections, both as reference and parent, are removed from the dataframe.
       - The function specifically renames certain index values that correspond to internal and section-end-external segments.
   """
    # Select external iax connections (between parent and children nodes)
    # these are just references to the original dataframe, not making copies
    df_segment_ref = df[df.index.get_level_values('ref').str.startswith(f'{section}(')]  # select iax rows where segment is the reference
    df_segment_par = df[df.index.get_level_values('par').str.startswith(f'{section}(')]  # select iax rows where segment is the parent
    # this is now a new copy - so renaming does not affects the original dataframe
    df_external = pd.concat([df_segment_ref, df_segment_par]).drop_duplicates(keep=False)  # this keeps rows that are unique (meaning that they connect to external nodes)

    # Rename index
    iref = df.index.get_level_values('ref').str.startswith(f'{section}(')
    first_internal_name = df.index.get_level_values('ref')[iref].sort_values()[0] # we assume that sorting will sort it correctly: The first element is the first internal node
    last_terminal_name = df.index.get_level_values('ref')[iref].sort_values()[-1] # and the last is the terminal node
    rename_dict = {first_internal_name: section,
                   last_terminal_name: section}
    df_external.rename(index=rename_dict, level='ref', inplace=True)
    df_external.rename(index=rename_dict, level='par', inplace=True)

    # Remove segment iax rows (both external and internal)
    df_internal_idx = pd.concat([df_segment_ref, df_segment_par]).drop_duplicates().index
    df.drop(df_internal_idx, inplace=True)

    # Concatenate updated external iax rows
    df_merged_dendritic_section = pd.concat([df, df_external])
    return df_merged_dendritic_section

def update_root_node(df_merged: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    Updates the root node in the given dataframe by switching the reference and parent segments along the shortest
    path between a new root and the original root ('soma'), and reversing the axial current (iax) values.

    This function modifies the reference-parent pairs and axial current values of the edges on the (shortest) path
    between the new root and the original root (soma), updating the dataframe accordingly.

    Args:
        df_merged : pd.DataFrame
            A dataframe containing axial current (iax) data with a multi-level index consisting of reference ('ref')
            and parent ('par') segments. The new root node should be represented by a section where the segment values
            have already been merged.
        section : str
            The section identifier representing the new root node.

    Returns:
        pd.DataFrame
            A new dataframe where the axial current connections along the shortest path between the new root and
            the original root ('soma') have been updated by switching the reference-parent pairs and negating the
            axial current values.

    Notes:
        - The reference and parent segments of the edges on the shortest path are switched, and the axial current values
          are multiplied by -1 to reflect the change in direction.
        - The resulting dataframe is re-indexed and returned, with the reference ('ref') and parent ('par') columns properly set.
    """
    # The input of this function should be a dataframe where the new root node is a section where the segment values are already merged
    dg = create_directed_graph(df_merged, df_merged.columns[0])
    g = dg.to_undirected()

    original_root = 'soma'
    new_root = section

    # Extract iax rows that are on the shortest path between the new root and the soma (original root)
    path = nx.shortest_path(g, source=new_root, target=original_root)  # select nodes of the shortest path between soma and new root
    edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]  # create node pairs for each edge in the path
    df_edges_in_path = df_merged[df_merged.index.isin(edges_in_path)]  # select iax rows of the path

    # Switch ref-par pairs and multiply iax values by -1
    df_switched = df_edges_in_path.copy()
    df_switched.index = pd.MultiIndex.from_tuples([(b, a) for a, b in df_edges_in_path.index])
    df_switched = -df_switched

    # Update original dataframe
    df_updated = df_merged.copy()
    df_updated = df_updated.drop(df_edges_in_path.index)  # drop rows corresponding to the original node pairs in the path
    df_updated = pd.concat([df_updated, df_switched])
    df_updated = df_updated.reset_index()
    df_updated = df_updated.rename(columns={'level_0': 'ref', 'level_1': 'par'})
    df_updated = df_updated.set_index(['ref', 'par'])
    return df_updated

def create_region_specific_index(df: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    """
    Creates region-specific index by mapping each segment to a predefined region
    and categorizing intrinsic and synaptic types.

    Args:
        df : pd.DataFrame
            The dataframe containing the original multi-indexed data with 'segment' and 'itype' columns

        input_dir : str
            The directory containing text files corresponding to different regions.
            Each file should have a list of segment names associated with that region.

    Returns:
        pd.DataFrame
            A new DataFrame with:
            - 'segment': The original segment names.
            - 'itype': A combined label of the mapped region and current type (e.g., 'axon_intrinsic').

    Notes:
        - The function reads predefined region text files and maps segments to their respective regions.
        - If a segment is not found in any region list, it is labeled as 'Unknown'.
        - The function also categorizes current types as either 'intrinsic' or 'synaptic'.
        - The final 'itype' column is a combination of the detected region and type.
    """

    fnames_regions = ['distal', 'oblique_trunk', 'axon', 'basal', 'soma']

    # Create a dictionary with each region's file contents split by newline
    regions_dict = {}
    for f in fnames_regions:
        with open(os.path.join(input_dir, f + '.txt'), 'r') as file:
            contents = file.read().strip()  # Read the file and strip leading/trailing whitespace
            segments = contents.split('\n')  # Split the contents by newline
            key = os.path.splitext(f)[0]  # Use the file name (without extension) as the key
            regions_dict[key] = segments  # Store the list of segments as the value

    region_values = []
    segments = df['segment'].values
    for segment in segments:
        seg = segment.split('(')[0].strip()  # Clean the segment
        region_value = 'Unknown'
        # Check if the segment exists in any of the lists in regions_dict
        for key, value in regions_dict.items():
            if seg in value:  # Check if the cleaned segment exists in the list of segments
                region_value = key  # Assign the corresponding region key
                break
        region_values.append(region_value)

    # Create a dictionary that categorizes current types
    type_dict = {'intrinsic':['capacitive', 'car', 'kad', 'kap', 'kdr', 'kslow', 'nad', 'nax', 'passive'],
                 'synaptic':['AMPA', 'GABA', 'GABA_B', 'NMDA']}

    type_values = []
    types = df['itype'].values
    for type in types:
        type_value = 'Unknown'
        for key, value in type_dict.items():
            if type in value:
                type_value = key
                break
        type_values.append(type_value)

    # Combine region and current type labels
    combined_region_and_type = []
    for i, region_value in enumerate(region_values):
        combined_region_and_type.append(f'{region_value}_{type_values[i]}')

    # Create dataframe that contains the region-specific multiindex
    region_specific_index = pd.DataFrame()
    region_specific_index['segment'] = df['segment']
    region_specific_index['itype'] = combined_region_and_type
    return region_specific_index


def calc_im_by_region(df: pd.DataFrame) -> pd.DataFrame:
    # Extract unique segments and currents (itypes)
    segments = df.index.get_level_values(0).unique()
    currents = df.index.get_level_values(1).unique()

    # Create a MultiIndex from the unique values of segment and itype
    multi_index = pd.MultiIndex.from_product([segments, currents], names=['segment', 'itype'])

    # Create a new DataFrame with the MultiIndex and numerical columns initialized to zero
    df_new = pd.DataFrame(index=multi_index, columns=df.columns)
    df_new[:] = 0.0  # initialize numerical values to zero

    # Loop through each segment and group by itype, summing the numerical columns
    for seg in segments:
        # Group by 'itype' and sum the numerical columns for that segment
        df_seg_grouped = df.loc[seg].groupby('itype').sum()

        # Update df_new with the grouped results
        for itype, row in df_seg_grouped.iterrows():
            df_new.loc[(seg, itype), :] = row
    df_new = df_new.astype(np.float32)
    return df_new

def partition_iax_single(ref: str, par: str, tp: int, im_signed: pd.DataFrame, iax_tp: float) -> None:
    """
    Partitions axial currents at a specific time point into membrane currents and updates the parent node's membrane currents.

    Parameters:
        ref (str): The reference node (child node) in the current partitioning process.
        par (str): The parent node in the current partitioning process.
        tp (int): The time point for which the partitioning is performed.
        im (DataFrame): A DataFrame containing membrane currents for each node and time point.
        iax (DataFrame): A DataFrame containing axial currents for each reference-parent pair and time point.

    Returns:
        None: The function updates the `im` DataFrame in place with the partitioned axial currents added to the parent node's membrane currents.
    """
    im_tp = im_signed.loc[ref, tp]
    # iax_tp either POSITIVE or NEGATIVE
    part_curr = im_tp / im_tp.sum() * iax_tp
    updated_curr = im_signed.loc[par, tp] + part_curr
    im_signed.loc[par, tp] = updated_curr.values.astype(np.float32)  # update original dataframe of the membrane currents
