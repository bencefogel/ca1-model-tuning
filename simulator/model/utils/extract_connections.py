import pandas as pd
import networkx as nx
from neuron import h


def get_external_connections():
    ref_external = []
    par_external = []
    ri_external = []

    for sec in h.allsec():
        counter = 0
        for seg in sec:
            if counter < 1:
                ref_external.append(seg)
                par_external.append(sec.parentseg())
                try:
                    ri_external.append(seg.ri())
                    counter += 1
                except:
                    print(f"The following section does not have a parent: {sec}")
                    ri_external.append(-1)
                    counter += 1
                    continue

    connections_external = pd.DataFrame()
    connections_external["ref"] = ref_external
    connections_external["par"] = par_external
    connections_external["ri"] = ri_external
    connections_external['ref'] = connections_external['ref'].astype(str)
    connections_external['par'] = connections_external['par'].astype(str)
    connections_external['ri'] = connections_external['ri'].astype(float)

    return connections_external


def get_internal_connections():
    ref_internal = []
    par_internal = []
    ri_internal = []

    for sec in h.allsec():
        segments = []
        for seg in sec.allseg():
            segments.append(seg)

        for i, _ in enumerate(segments):
            if i > 0:
                ref_internal.append(segments[i])
                par_internal.append(segments[i - 1])
                ri_internal.append(segments[i].ri())

    connections_internal = pd.DataFrame()
    connections_internal["ref"] = ref_internal
    connections_internal["par"] = par_internal
    connections_internal["ri"] = ri_internal
    connections_internal['ref'] = connections_internal['ref'].astype(str)
    connections_internal['par'] = connections_internal['par'].astype(str)
    connections_internal['ri'] = connections_internal['ri'].astype(float)
    return connections_internal


def get_connections(connections_external, connections_internal):
    connections = connections_internal.copy()
    for index, row in connections_external.iterrows():
        ref = row["ref"]
        par = row["par"]
        ri = row["ri"]

        connections.loc[connections["ref"] == ref, "par"] = par
        connections.loc[connections["ref"] == ref, "ri"] = ri

    # the following 3 lines are for sanity check
    # G = nx.from_pandas_edgelist(connections, source='par', target='ref')
    # connected_components = list(nx.connected_components(G))
    # print(f"Number of graphs originally: {len(connected_components)}")

    # issue 1: dend2_0(0.5),soma(0.709232),0.15333 -> this is connected to soma(0.833333) in practice
    # issue 2: dend3_0(0.5),soma(0.443475),0.05747 -> this is connected to soma(0.5) in practice
    connections.loc[connections["ref"] == "dend2_0(0.5)", "par"] = "soma(0.833333)"
    connections.loc[connections["ref"] == "dend3_0(0.5)", "par"] = "soma(0.5)"

    # the following 3 lines are for sanity check
    # G_corrected = nx.from_pandas_edgelist(connections, source='par', target='ref')
    # connected_components_corrected = list(nx.connected_components(G_corrected))
    # print(f"Number of graph components after correction: {len(connected_components_corrected)}")
    return connections