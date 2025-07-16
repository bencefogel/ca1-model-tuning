from neuron import h
import sys
import numpy as np

h('objref nil')

modpath = 'simulator/model/density_mechs'
h.nrn_load_dll(modpath + '\\nrnmech.dll')


def init_activeCA1(model):
    model.soma.insert('nax')
    model.soma.gbar_nax = model.gna_soma
    model.soma.insert('kdr')
    model.soma.gkdrbar_kdr = model.gkdr_soma
    model.soma.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.hill.insert('nax')
    model.hill.gbar_nax = model.gna_axon
    model.hill.insert('kdr')
    model.hill.gkdrbar_kdr = model.gkdr_axon
    model.soma.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.iseg.insert('nax')
    model.iseg.gbar_nax = model.gna_axon
    model.iseg.insert('kdr')
    model.iseg.gkdrbar_kdr = model.gkdr_axon
    model.iseg.insert('kap')
    model.soma.gkabar_kap = model.gka

    model.node[0].insert('nax')
    model.node[0].gbar_nax = model.gna_node
    model.node[0].insert('kdr')
    model.node[0].gkdrbar_kdr = model.gkdr_axon
    model.node[0].insert('kap')
    model.node[0].gkabar_kap = model.gka * 0.2

    model.node[1].insert('nax')
    model.node[1].gbar_nax = model.gna_node
    model.node[1].insert('kdr')
    model.node[1].gkdrbar_kdr = model.gkdr_axon
    model.node[1].insert('kap')
    model.node[1].gkabar_kap = model.gka * 0.2

    model.inode[0].insert('nax')
    model.inode[0].gbar_nax = model.gna_axon
    model.inode[0].insert('kdr')
    model.inode[0].gkdrbar_kdr = model.gkdr_axon
    model.inode[0].insert('kap')
    model.inode[0].gkabar_kap = model.gka * 0.2

    model.inode[1].insert('nax')
    model.inode[1].gbar_nax = model.gna_axon
    model.inode[1].insert('kdr')
    model.inode[1].gkdrbar_kdr = model.gkdr_axon
    model.inode[1].insert('kap')
    model.inode[1].gkabar_kap = model.gka * 0.2

    model.inode[2].insert('nax')
    model.inode[2].gbar_nax = model.gna_axon
    model.inode[2].insert('kdr')
    model.inode[2].gkdrbar_kdr = model.gkdr_axon
    model.inode[2].insert('kap')
    model.inode[2].gkabar_kap = model.gka * 0.2

    for d in model.dends:
        d.insert('nad')
        d.gbar_nad = model.gna
        d.insert('kdr')
        d.gkdrbar_kdr = model.gkdr
        d.insert('kap')
        d.gkabar_kap = 0
        d.insert('kad')
        d.gkabar_kad = 0

    h('access soma')
    h('distance()')

    ## for the apicals: KA-type depends on distance
    ## density is as in terminal branches - independent of the distance
    for sec in h.all_apicals:
        nseg = sec.nseg
        iseg = 0
        for seg in sec:
            xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
            xdist = h.distance(xx, sec=sec)
            if (xdist > model.dprox):
                seg.gkabar_kad = model.gka
            else:
                seg.gkabar_kap = model.gka
            iseg = iseg + 1

    h('access soma')
    h('distance()')

    ## distance dependent A-channel densities in apical trunk dendrites
    ##      1. densities increase till 'dlimit' with dslope
    ##      2. proximal channels switch to distal at 'dprox'
    ##      3. sodium channel density also increases with distance
    for sec in h.primary_apical_list:
        nseg = sec.nseg
        sec.insert('nax')
        iseg = 0
        for seg in sec:
            # 0. calculate the distance from soma
            xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
            xdist = h.distance(xx, sec=sec)
            # 1. densities increase till 'dlimit' with dslope
            if (xdist > model.dlimit):
                xdist = model.dlimit
            # 2. proximal channels switch to distal at 'dprox'
            if (xdist > model.dprox):
                seg.gkabar_kad = model.gka_trunk * (1 + xdist * model.dslope)
            else:
                seg.gkabar_kap = model.gka_trunk * (1 + xdist * model.dslope)
            iseg = iseg + 1
            # 3. sodiom channel density also increases with distance
            if (xdist > model.nalimit):
                xdist = model.nalimit
            seg.gbar_nax = model.gna_trunk * (1 + xdist * model.naslope)

            seg.gbar_nad = 0
            seg.gkdrbar_kdr = model.gkdr_trunk

    ## for the basals: all express proximal KA-type
    ## density does not increase with the distance
    for sec in h.all_basals:
        for seg in sec:
            seg.gkabar_kap = model.gka

    # Adding Ca and K_slow conductances
    for sec in h.all_apicals:
        sec.insert('car')
        sec.gmax_car = 0.006  # 0.007
        sec.insert('kslow')
        sec.gmax_kslow = 0.001  # 0.003


def addClustLocs(model, nsyn, Nclust, Ncell_per_clust, seed, midle=False, clocs=None, Lmin=60):
    # Nclust clusters in a random background innervation
    # clusters are ADDED to the random background
    # clustering:
    #     1. select pre clusters by genClustStarts
    #     2. generate random Elocs
    #     3. select post cluster locations by genClusts
    #     4. add post cluster locations
    # output: Elocs: list with [dend id, location]

    # 1. presynaptic partners belonging to the clusters
    if midle == True:  # clusters start at the middle of the maze
        Nsyn_in_clust = Nclust * Ncell_per_clust  # number of synapses in clusters
        Nsyn_rand = nsyn - Nsyn_in_clust
        istart = 880  # int(Nsyn_rand / 2)
        iend = 1120  # int(Nsyn_rand / 2) + Nsyn_in_clust
        # istart = int(Nsyn_rand / 2)
        # iend = int(Nsyn_rand / 2) + Nsyn_in_clust
        prestarts = np.arange(istart, iend, Ncell_per_clust)

    # index of pre cells IN clusters
    ind_clustpre = np.arange(prestarts[0], prestarts[0] + Ncell_per_clust)
    if Nclust > 0:
        for j in np.arange(1, Nclust):
            istart = prestarts[j]
            iend = prestarts[j] + Ncell_per_clust
            ind_clustpre = np.concatenate((ind_clustpre, np.arange(istart, iend)))

    # 2. random locations - background synapses
    if (Nsyn_rand):
        bg_Elocs = genRandomLocs(model, Nsyn_rand, seed=100 * seed + 1)
        np.random.seed(100 * seed + 2)
        np.random.shuffle(bg_Elocs)
    else:
        bg_Elocs = []

    # 3. clustered locations
    # def genClusts(Nclust, Ncell_per_clust, minL, seed):
    clustered_Elocs, clDends = genClusts(model, Nclust, Ncell_per_clust, Lmin, seed=100 * seed + 4, clocs=clocs)
    # clustered_Elocs = genClusts(Nclust, Ncell_per_clust, 1, seed=100*seed+4)

    # 4. cycle through all clusters and add the post from either the
    #       random or the post cluster locations
    j = 0  # POST - clustered
    k = 0  # POST - random
    Elocs = bg_Elocs + clustered_Elocs
    for i in np.arange(nsyn):  # PRE
        if i in ind_clustpre:  # take it from the cluster - list
            Elocs[i] = clustered_Elocs[j]
            j = j + 1
        else:
            Elocs[i] = bg_Elocs[k]
            k = k + 1
    return Elocs, ind_clustpre, clDends


def genRandomLocs(model, nsyn, seed, dend_ids=None):
    # randomly choose a dendritic branch, proportionally to its length
    np.random.seed(seed)
    nden = len(model.dends)
    Ldends = np.zeros(nden)
    for ii in np.arange(0, nden): Ldends[ii] = model.dends[ii].L

    Pdends = Ldends / sum(Ldends)
    if dend_ids is not None:
        Pdends = np.zeros(nden)
        for dend in dend_ids:
            Pdends[dend] = Ldends[dend] / sum(Ldends)
        Pdends = Pdends / sum(Pdends)

    Ndends = np.random.multinomial(nsyn, Pdends)

    #  and insert a nsyn synapses at random locations within the branch
    locs = []
    for dend in np.arange(0, nden):  #
        nsyn_dend = Ndends[dend]
        if (nsyn_dend > 0):
            locs_dend = np.sort(np.random.uniform(0, 1, nsyn_dend))
            for s in np.arange(0, nsyn_dend):
                locs.append([dend, locs_dend[s]])

    return locs


def genClusts(model, Nclust, Ncell_per_clust, minL, seed, clocs=None):
    # randomly choose a dendritic branch, larger than minL for clustered synapses
    # clocs:
    if (minL < Ncell_per_clust):
        print('error: cluster size mismatch, stop simulation! Number of cells per cluster (1 um inter-spine distance):',
              Ncell_per_clust, ', minL:', minL)
        sys.exit(1)
    np.random.seed(seed)
    nden = len(model.dends)
    Ldends = np.zeros(nden)
    for ii in np.arange(0, nden):
        Ldends[ii] = model.dends[ii].L
    idends = np.flatnonzero(Ldends > minL)

    replace_clusts = False
    if (Nclust > len(idends)):
        print('warning: cluster number mismatch, multiple clusters are allocated to the same branch! Nclust:', Nclust,
              ', number of available branches:', len(idends), ', minL:', minL)
        replace_clusts = True

    # if some dendrites are preselected ...
    if clocs is not None:
        k = len(clocs)
        clDends1 = clocs  # we choose those dendrites
        if (Nclust > k):  # we choose randomly from the others
            if (replace_clusts == False):
                idends = np.setdiff1d(idends, clocs)
            clDends2 = np.random.choice(idends, Nclust - k, replace=replace_clusts)
            clDends = np.concatenate((clDends1, clDends2))
        else:
            clDends = np.array(clDends1)

    else:
        clDends = np.random.choice(idends, Nclust, replace=replace_clusts)
    #  and insert a nsyn synapses at random locations within the branch

    locs = []
    i_dend = 0
    for dend in clDends:
        dend = int(dend)
        locstart = np.random.uniform(0, 1 - Ncell_per_clust / Ldends[dend], 1)
        if clocs is not None:
            if i_dend < k:
                locstart = (1 - Ncell_per_clust / Ldends[dend]) / 2
        for s in np.arange(0, Ncell_per_clust):
            locs.append([dend, (s / Ldends[dend] + locstart).item()])  # 1 um distance between spines
        i_dend = i_dend + 1
    return locs, clDends


def add_syns(model, Elocs, Ilocs):
    model.AMPAlist = []
    model.ncAMPAlist = []
    AMPA_gmax = 0.6 / 1000.  # Set in nS and convert to muS data.Agmax

    model.NMDAlist = []
    model.ncNMDAlist = []
    NMDA_gmax = 0.8 / 1000.  # Set in nS and convert to muS data.Ngmax

    for loc in Elocs:
        locInd = int(loc[0])
        if (locInd == -1):
            synloc = model.soma
        else:
            synloc = model.dends[int(loc[0])]
            synpos = float(loc[1])

        AMPA = h.Exp2Syn(synpos, sec=synloc)
        AMPA.tau1 = 0.1  # data.Atau1
        AMPA.tau2 = 1  # data.Atau2
        NC = h.NetCon(h.nil, AMPA, 0, 0, AMPA_gmax)  # NetCon(source, target, threshold, delay, weight)
        model.AMPAlist.append(AMPA)
        model.ncAMPAlist.append(NC)

        NMDA = h.Exp2SynNMDA(synpos, sec=synloc)
        NMDA.tau1 = 2  # data.Ntau1
        NMDA.tau2 = 50  # data.Ntau2
        NC = h.NetCon(h.nil, NMDA, 0, 0, NMDA_gmax)
        x = float(loc[1])
        model.NMDAlist.append(NMDA)
        model.ncNMDAlist.append(NC)

    model.GABAlist = []
    model.ncGABAlist = []
    GABA_gmax = 0.2 / 1000.  # Set in nS and convert to muS data.Igmax 0.1/1000

    model.GABA_Blist = []
    model.ncGABA_Blist = []
    GABAB_gmax = 0.2 / 1000.  # Set in nS and convert to muS data.Bgmax 0.1/1000

    for loc in Ilocs:
        locInd = int(loc[0])
        if (locInd == -1):
            synloc = model.soma
        else:
            synloc = model.dends[int(loc[0])]
        GABA = h.Exp2Syn(float(loc[1]), sec=synloc)
        GABA.tau1 = 0.1  # data.Itau1
        GABA.tau2 = 4  # data.Itau2
        GABA.e = -65  # data.Irev
        NC = h.NetCon(h.nil, GABA, 0, 0, GABA_gmax)
        model.GABAlist.append(GABA)
        model.ncGABAlist.append(NC)

        GABAB = h.Exp2Syn(float(loc[1]), sec=synloc)
        GABAB.tau1 = 1  # data.Btau1
        GABAB.tau2 = 40  # data.Btau2
        GABAB.e = -80  # data.Brev
        NC = h.NetCon(h.nil, GABAB, 0, 0, GABAB_gmax)
        model.GABA_Blist.append(GABAB)
        model.ncGABA_Blist.append(NC)
