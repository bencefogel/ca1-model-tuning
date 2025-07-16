from neuron import h
import numpy as np

## define the CA1 cell
class CA1(object):

    def __init__(self):
        h('xopen("simulator/model/CA1.hoc")')
        propsCA1(self)
        # self._geom()
        self._topol()
        self._biophys()

    def _topol(self):
        self.soma = h.soma
        self.hill = h.hill
        self.iseg = h.iseg
        self.node = h.node
        self.inode = h.inode
        self.dends = []
        for sec in h.allsec():
            self.dends.append(sec)
            n_seg = int(max(1, round(sec.L / self.seclength + 0.5)))
            if ((n_seg - int(n_seg/2)*2)==0) :
                n_seg = n_seg + 1
            sec.nseg = n_seg
        for i in np.arange(8):
            self.dends.pop()   # Remove soma and axons from the list
        # self.axon.connect(self.soma,1,0)


    def _biophys(self):
        for sec in h.allsec():
            sec.cm = self.CM
            sec.insert('pas')
            sec.e_pas = self.E_PAS
            sec.g_pas = 1.0/self.RM
            sec.Ra = self.RA

        h.soma.g_pas = 1.0 /self.RM_soma

        h.node[0].g_pas = 1.0 /self.RM_node
        h.node[0].cm = self.CM

        h.node[1].g_pas = 1.0 /self.RM_node
        h.node[1].cm = self.CM

        h.inode[0].g_pas = 1.0 /self.RM
        h.inode[0].cm = self.CM_inode

        h.inode[1].g_pas = 1.0 /self.RM
        h.inode[1].cm = self.CM_inode

        h.inode[2].g_pas = 1.0 /self.RM
        h.inode[2].cm = self.CM_inode


        ## compensate for spines
        h('access soma')
        h('distance()')
        for sec in self.dends:
        # for sec in h.all_apicals: # in the Katz model compensation was done only for apicals.
            nseg = sec.nseg
            iseg = 0
            for seg in sec :
            # 0. calculate the distance from soma
                xx = iseg * 1.0/nseg + 1.0 / nseg / 2.0
                xdist=h.distance(xx, sec=sec)
                # print(sec.name(), xx, xdist)
                # 1. calculate the diameter of the segment
                xdiam=seg.diam
                # print(sec.name(), xx, xdiam)
                if ((xdist > self.spinelimit) + (xdiam < self.spinediamlimit)):
                    seg.cm = self.CM * self.spinefactor
                    seg.g_pas = self.spinefactor * 1.0/self.RM#_dend
                    # sec.Ra = self.RA_dend # Ra is a section variable...
                iseg = iseg + 1

def propsCA1(model):

   # Passive properties
    model.CELSIUS = 35
    model.v_init = -68.3 # -68.3 for theta and -72 for replay
    model.RA = 100 # 150.00           # internal resistivity in ohm-cm
    model.RA_dend = 100 # 200
    model.CM = 1 # 1                # 0.75     # specific membrane capacitance in uF/cm^2
    model.CM_inode=0.04           # capacitance in myelin

    model.RM = 20000 # was 3000 for hCond;  20000       # specific membrane resistivity at in ohm-cm^2
    model.RM_dend = 20000 # 10 000 - 40 000; only used in Robustness analysis in Adam's paper
    model.RM_soma = 40000 # was 3000 for hCond;  20000       # specific membrane resistivity at the soma in ohm-cm^2
    model.RM_inode = 40000 #200000          # inter-nodal resistivity with myelin
    model.RM_node = 50 #200000          # nodal resistivity

    model.E_PAS = -66 # -66 - set to v_init if passive
    model.seclength = 10      # um, the length of a section
    model.spinefactor = 2       # 2 factor by which to change passive properties
    model.spinelimit = 100      # 100 distance beyond which to modify passive properties to account for spines
    model.spinediamlimit = 1      # 100 distance beyond which to modify passive properties to account for spines

   # Active properties - Values from the Spruston-lab (Katz et al., 2009) fitted only to trunk data!
    model.gna = 0.03      # 0.03; 0.01 sodium conductance in terminal branches
    model.gna_trunk = 0.04      # 0.04 sodium conductance
    model.gna_axon = 0.04      # 0.04 sodium conductance in the axon
    model.gna_soma = 0.2    # 0.04 - 0.2 sodium conductance in the soma
    model.gna_node = 15      # 30 - 15 sodium conductance in the axon
    model.nalimit = 500
    model.naslope = 0.001       # 0.001 is 'strong' propagation on the trunk
    model.gna_dend_hotSpot = 5

    model.gkdr = 0.02        # 0.005 delayed rectifier density in terminal branches
    model.gkdr_trunk = 0.040        # 0.04 delayed rectifier density in the trunk
    model.gkdr_soma = 0.04        # 0.04 delayed rectifier density at the soma
    model.gkdr_axon = 0.04        # 0.04 delayed rectifier density at the axon

    model.gka = model.gkdr      # 0.005 A-type potassium density in terminal branches
    model.gka_trunk = 0.048      # 0.048  A-type potassium starting density in the trunk
    model.dlimit = 500        # cut-off for increase of A-type density
    model.dprox = 100          # distance to switch from proximal to distal type
    model.dslope=0.01         # slope of A-type density

#    model.gna = 0 #in original code modulateNa = True


