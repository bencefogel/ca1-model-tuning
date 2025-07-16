import pandas as pd
import networkx as nx


def get_segment_iax(segment, df):
    """
    Returns a DataFrame containing the axial current values for a specific segment.

    This function extracts the axial currents associated with a
    specified segment. It handles both cases where the segment is a reference (ref)
    or a parent (par) in the provided MultiIndex DataFrame.

    Parameters:
        segment (str): The name of the segment whose axial currents are to be extracted.
        df (pd.DataFrame): A DataFrame with a MultiIndex containing 'ref' and 'par'
                           levels, and values representing axial currents.

    Returns:
        pd.DataFrame: A DataFrame containing the axial currents for the given segment.
    """
    ref_mask = df.index.get_level_values("ref") == segment
    ref_iax = -1 * df[ref_mask]  # negated, because axial currents are calculated from the perspective of the parent node

    par_mask = df.index.get_level_values("par") == segment
    par_iax = df[par_mask]

    df_iax_seg = pd.concat([ref_iax, par_iax], axis=0)

    return df_iax_seg


def update_root_node(df_merged: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    Updates the root node in the given dataframe by switching the reference and parent segments along the shortest
    path between a new root and the original root ('soma'), and reversing the axial current (iax) values.

    This function modifies the reference-parent pairs and axial current values of the edges on the (shortest) path
    between the new root and the original root (soma), updating the dataframe accordingly.

    Parameters:
    ----------
    df_merged : pd.DataFrame
        A dataframe containing axial current (iax) data with a multi-level index consisting of reference ('ref')
        and parent ('par') segments. The new root node should be represented by a section where the segment values
        have already been merged.
    section : str
        The section identifier representing the new root node.

    Returns:
    -------
    pd.DataFrame
        A new dataframe where the axial current connections along the shortest path between the new root and
        the original root ('soma') have been updated by switching the reference-parent pairs and negating the
        axial current values.

    Notes:
    ------
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


def create_directed_graph(iax: pd.DataFrame, tp: int) -> nx.DiGraph:
    """
    Creates a directed graph based on the axial current value (iax) at a specified time point (tp).

    Parameters:
        iax (df): A pandas DataFrame with axial current data. It must include a column for the specified time point (`tp`)
                            and index columns representing 'ref' and 'par' segments.
        tp (int): The time point for which to construct the graph using axial current values.

    Returns:
        DiGraph: A directed graph where edges are added based on the sign of the axial current values.
                    - If `iax` is positive, the edge direction is `par -> ref`.
                    - If `iax` is negative, the edge direction is `ref -> par`.
    """
    df_iax_tp = iax[tp]
    df_iax_tp = df_iax_tp.reset_index()
    df_iax_tp.rename(columns={tp: "iax"}, inplace=True)  # has three columns: ref, par, iax

    # Create directed graph (add edges to the graph based on the sign of iax)
    dg = nx.DiGraph()
    for index, row in df_iax_tp.iterrows():
        if row['iax'] >= 0:
            dg.add_edge(row['par'], row['ref'], iax=row['iax'])  # par -> ref if 'iax_timepoint' is positive
        elif row['iax'] < 0:
            dg.add_edge(row['ref'], row['par'], iax=row['iax'])  # ref -> par if 'iax_timepoint' is negative
    return dg