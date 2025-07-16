import networkx as nx
import pandas as pd


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


def get_partitioning_order(dg: nx.DiGraph, target: str, direction: str) -> list[tuple[str, str]]:
    """
    Determines the partitioning order of nodes in a directed graph.

    Parameters:
        dg (DiGraph): The directed graph.
        target (str): The node from which the traversal starts.
        direction (str): The traversal direction. Options are:
                         - "out": Outward traversal from the target node.
                         - "in": Inward traversal towards the target node.

    Returns:
        list[tuple[str, str]]: A list of node pairs representing the traversal order.
    """
    traversal_methods = {
        "out": get_traversal_order_out,
        "in": get_traversal_order_in
    }
    return traversal_methods[direction](dg, target)


def get_traversal_order_out(dg: nx.DiGraph, target: str) -> list[tuple[str, str]]:
    """
    Computes the outward traversal order from a target node in a directed graph.

    Parameters:
        dg (DiGraph): The directed graph.
        target (str): The node from which the outward traversal starts.

    Returns:
        list[tuple[str, str]]: A list of node pairs representing the traversal order,
                               starting from the leaf nodes.
    """
    # Find subgraph using depth first search algorithm and copy iax values of edges
    dg_dfs_out = nx.dfs_tree(dg, source=target)
    for u, v in dg_dfs_out.edges():
        if dg.has_edge(u, v):
            dg_dfs_out[u][v]['iax'] = dg[u][v]['iax']

    # Extract traversal order starting from the leaf nodes
    edges_visited_out = list(nx.edge_dfs(dg_dfs_out, target))
    node_pairs_out = [(v, u) for (u, v) in edges_visited_out]  # switch nodes of each edge
    node_pairs_out.reverse()  # reverse node pairs order (to start from the leaf nodes)
    return node_pairs_out


def get_traversal_order_in(dg: nx.DiGraph, target: str) -> list[tuple[str, str]]:
    """
    Computes the inward traversal order towards a target node in a directed graph.

    Parameters:
        dg (nx.DiGraph): The directed graph.
        target (str): The node towards which the inward traversal is computed.

    Returns:
        list[tuple[str, str]]: A list of node pairs representing the traversal order,
                               starting from the leaf nodes.
    """
    dg_reversed = nx.reverse(dg, copy=True)
    # Find subgraph using depth first search algorithm and copy iax values of edges
    dg_dfs_in = nx.dfs_tree(dg_reversed, source=target)
    for u, v in dg_dfs_in.edges():
        if dg.has_edge(v, u):
            dg_dfs_in[u][v]['iax'] = dg[v][u]['iax']

    # Extract traversal order starting from the leaf nodes
    edges_visited_in = list(nx.edge_dfs(dg_dfs_in, target))
    node_pairs_in = [(v, u) for (u, v) in edges_visited_in]  # switch nodes of each edge
    node_pairs_in.reverse()  # reverse node pairs order (to start from the leaf nodes)
    return node_pairs_in
