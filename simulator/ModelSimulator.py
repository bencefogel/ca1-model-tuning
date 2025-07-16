import pandas as pd
import numpy as np
from neuron import h

import simulator.model.saveClass as sc
import simulator.model.simulation as simulation
from simulator.model.ca1_model import CA1
from simulator.model.ca1_functions import init_activeCA1, addClustLocs, genRandomLocs, add_syns
from simulator.model.sim_functions import sim_PlaceInput
from simulator.model.utils.extract_connections import get_external_connections, get_internal_connections, get_connections
from simulator.model.utils.extract_areas import get_segment_areas

class ModelSimulator:
    """
    ModelSimulator is a class that constructs a CA1 hippocampal model, configures synaptic inputs,
    and runs simulations. It manages both the model building process and the
    execution of simulations, returning the simulation data including membrane potentials, membrane currents,
    segment connections and segment areas.
    """

    def __init__(self):
        """
       Initialize the ModelSimulator object.

       Args:
           connections (dict): Stores internal and external segment connections for later processing.
           segment_areas (pd.DataFrame): DataFrame containing segment name and segment area information.
       """
        self.connections = {}
        self.segment_areas = pd.DataFrame()

    def build_model(self, cluster_seed: int, random_seed: int) -> CA1:
        """
        Builds the CA1 model with synaptic connections.

        Args:
            cluster_seed (int): Seed for clustering excitatory synapses.
            random_seed (int): Seed for random synapse location generation.

        Returns:
            model (CA1): An instance of the CA1 model with configured synapses.
        """
        print("Building CA1 model...")
        # Initialize model
        model = CA1()
        init_activeCA1(model)

        # Generate excitatory synapse locations and cluster locations
        Elocs, ind_clust, clDends = addClustLocs(
            model, nsyn=2000, Nclust=12, Ncell_per_clust=20,
            seed=random_seed, midle=True, clocs=[], Lmin=60
        )

        clocs = np.zeros((16, 12))
        clocs[cluster_seed, :] = clDends

        # Generate inhibitory synapse locations
        Isomalocs = [[-1, 0.5] for _ in range(80)]
        np.random.seed(10001)
        Idendlocs = genRandomLocs(model, int(200 - 80), 10001)
        np.random.shuffle(Idendlocs)
        np.random.shuffle(Isomalocs)
        Ilocs = Isomalocs + Idendlocs

        # Add synapses to model
        add_syns(model, Elocs, Ilocs)

        # Modify clustered excitatory synapses
        for syn_id in ind_clust:
            model.ncAMPAlist[syn_id].weight[0] = 0.0010
            model.ncNMDAlist[syn_id].weight[0] = 0.0012

        # Get section and segment connections
        self.connections['external'] = get_external_connections()
        self.connections['internal'] = get_internal_connections()
        self.segment_areas = get_segment_areas()
        return model

    def run_simulation(self, model: CA1, e_input: str, i_input: str, simulation_time: float) -> dict:
        """
        Runs a simulation on the provided CA1 model with specified input files
        and parameters. Returns the simulation data.

        Args:
            model (CA1): The CA1 model instance.
            e_input (str): Path to the excitatory input file.
            i_input (str): Path to the inhibitory input file.
            simulation_time (float): Duration of the simulation in milliseconds.

        Returns:
            dict: The simulation data including membrane potential, membrane currents,
            segment connections and segment areas.
        """
        print("Running simulation...")
        simulation_data = sim_PlaceInput(model, Insyn=200, Irate=7.4, e_fname=e_input, i_fname=i_input,
                                         tstop=simulation_time, elimIspike=False)
        simulation_data['connections'] = get_connections(self.connections['external'], self.connections['internal'])
        simulation_data['areas'] = self.segment_areas
        return simulation_data
