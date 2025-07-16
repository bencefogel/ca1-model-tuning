import numpy as np

import simulator.model.simulation as simulation
from numpy import loadtxt

def sim_PlaceInput(model, Insyn, Irate, e_fname, i_fname, tstop, elimIspike=0): #sim_time in msec
    eitimes = readTrain(e_fname, i_fname)

    if ((Insyn > 0) & (Irate > 0)) :
        etimes = eitimes[0]
        itimes = eitimes[1]
        if (elimIspike > 0):
            N_ispikes = len(itimes)
            i_index = np.sort(np.random.choice(N_ispikes, int(round((1-elimIspike) * N_ispikes)), replace=False))
            itimes = itimes[i_index]

    # Run
    fih = simulation.h.FInitializeHandler(1, lambda: initSpikes(model, etimes, itimes))
    simulation_data = simulation.simulate(model, tstop)
    return  simulation_data

def initSpikes(model, etimes, itimes):
    if (len(etimes)>0):
        for s in etimes:
            model.ncAMPAlist[int(s[0])].event(float(s[1]))
            model.ncNMDAlist[int(s[0])].event(float(s[1]))

    if (len(itimes)>0):
        for s in itimes:
            model.ncGABAlist[int(s[0])].event(float(s[1]))
            model.ncGABA_Blist[int(s[0])].event(float(s[1]))

def readTrain(e_fname, i_fname):
    fname = e_fname
    Etimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)

    fname = i_fname
    Itimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)

    times = [Etimes, Itimes]
    return times

