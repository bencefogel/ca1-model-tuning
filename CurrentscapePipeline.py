import os

import numpy as np
import pandas as pd
from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from simulator.ModelSimulator import ModelSimulator
from preprocessor.Preprocessor import Preprocessor
from currentscape_visualization.currentscape import plot_currentscape


class CurrentscapePipeline:
    """
    A pipeline to simulate a neuron model, preprocess the membrane and axial currents,
    and save the results in chunks for use with Currentscape visualization.

    Attributes:
        output_dir (str): Directory to save all output files.
        cluster_seed (int): Random seed for clustering.
        random_seed (int): Random seed for synapse placement.
        e_input (str): Path to excitatory input spike file.
        i_input (str): Path to inhibitory input spike file.
        simulation_time (float): Total simulation time in ms.
        target (str): Target compartment for currentscape calculation.
        partitioning_strategy (str): Strategy for partitioning the model ('type' or 'region').
        filename (str): Name of the output PDF file for the generated currentscape plot.
        tmin (int): Start time (in ms) of the window to include in the preprocessing and currentscape plot.
        tmax (int): End time (in ms) of the window to include in the preprocessing and currentscape plot.
        simulation_data (dict): Dictionary holding the results of the simulation.
        taxis (np.ndarray): Time axis vector for the simulation.
    """

    def __init__(self,
                 output_dir: str, cluster_seed: int, random_seed: int, e_input: str, i_input: str, simulation_time: float,
                 target: str, partitioning_strategy: str, filename: str, tmin: int, tmax: int):

        self.output_dir = output_dir
        self.cluster_seed = cluster_seed
        self.random_seed = random_seed
        self.e_input = e_input
        self.i_input = i_input
        self.simulation_time = simulation_time
        self.target = target
        self.partitioning = partitioning_strategy
        self.currentscape_filename = filename
        self.tmin = tmin
        self.tmax = tmax
        self.simulation_data = None
        self.taxis = None


    @classmethod
    def from_preprocessed(cls, output_dir, target, partitioning_strategy, filename, tmin, tmax):
        """
        Alternative constructor for loading only preprocessed data.
        Skips parameters related to simulation.
        """
        return cls(
            output_dir=output_dir,
            cluster_seed=None,
            random_seed=None,
            e_input=None,
            i_input=None,
            simulation_time=None,
            target=target,
            partitioning_strategy=partitioning_strategy,
            filename=filename,
            tmin=tmin,
            tmax=tmax
        )


    def run_simulation(self):
        """
        Runs the simulation using the ModelSimulator.

        This method builds a neuron model, executes the simulation with the configured parameters, and stores
        the simulation data including the time axis ('taxis').
        """
        simulator = ModelSimulator()
        model = simulator.build_model(self.cluster_seed, self.random_seed)
        self.simulation_data = simulator.run_simulation(
            model,
            self.e_input,
            self.i_input,
            self.simulation_time
        )
        self.taxis = self.simulation_data['taxis']


    def preprocess(self):
        """
        Preprocesses simulation data for membrane and axial currents and saves
        the preprocessed data as CSV files.

        Args:
            simulation_data : dict
                The input simulation data to be processed.
            output_dir : str
                Path to the directory where the preprocessed files will be stored.
            im_path : str
                File path for the preprocessed membrane currents CSV.
            iax_path : str
                File path for the preprocessed axial currents CSV.
            im : DataFrame
                DataFrame containing the preprocessed membrane currents.
            iax : DataFrame
                DataFrame containing the preprocessed axial currents.
        """
        preprocessor = Preprocessor(self.simulation_data)
        self.im = preprocessor.preprocess_membrane_currents()
        self.iax = preprocessor.preprocess_axial_currents()

        pre_dir = os.path.join(self.output_dir, 'preprocessed')
        os.makedirs(pre_dir, exist_ok=True)
        self.im_path = os.path.join(pre_dir, 'im.csv')
        self.iax_path = os.path.join(pre_dir, 'iax.csv')

        self.im.to_csv(self.im_path)
        self.iax.to_csv(self.iax_path)
        np.save(os.path.join(self.output_dir, 'taxis.npy'), self.taxis)

        df_v = pd.DataFrame(self.simulation_data['membrane_potential_data'][1])
        df_v['segment'] = self.simulation_data['membrane_potential_data'][0]
        df_v.set_index('segment', inplace=True)
        df_v.to_csv(os.path.join(self.output_dir, 'vm.csv'))


    def calculate_currentscape(self):
        """
        Calculates the currentscape for a given target and partitioning strategy.

        This method creates an instance of the CurrentscapeCalculator class,
        specifies the directory containing the region list, and uses it to
        calculate the positive and negative partitioned currents based
        on the provided target, input files, and time constraints. The
        calculated values are stored in the attributes `part_pos` and
        `part_neg`.
        """
        region_list_dir = os.path.join('currentscape_calculator', 'region_list')
        calc = CurrentscapeCalculator(self.target, self.partitioning, region_list_dir)
        self.part_pos, self.part_neg = calc.calculate_currentscape(
            self.iax_path, self.im_path, self.taxis, self.tmin, self.tmax
        )

        res_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(res_dir, exist_ok=True)
        part_pos_path = os.path.join(res_dir, f'part_pos_{self.tmin}_{self.tmax}.csv')
        part_neg_path = os.path.join(res_dir, f'part_neg_{self.tmin}_{self.tmax}.csv')

        self.part_pos.to_csv(part_pos_path)
        self.part_neg.to_csv(part_neg_path)

    def visualize(self):
        """
        Generates a currentscape plot.

        This method processes the simulation data of membrane potential at a specific target,
        filters the time range, and plots the currentscape. It outputs the currentscape plot to a specified file.
        """
        self.taxis = np.load(os.path.join(self.output_dir, 'taxis.npy'))
        time_mask = (self.taxis > self.tmin) & (self.taxis < self.tmax)
        selected_indices = np.where(time_mask)[0]
        usecols_im = ['segment'] + list(map(str, selected_indices))

        df_v = pd.read_csv(os.path.join(self.output_dir, 'vm.csv'), usecols=usecols_im, index_col=0)
        df_v.columns = df_v.columns.astype(int)

        v_target = df_v.loc[f'{self.target}(0.5)'].values

        currentscape = plot_currentscape(
                                        self.part_pos, self.part_neg, v_target, self.taxis, self.tmin, self.tmax,
                                        return_segs=False, segments_preselected=False,
                                        partitionby=self.partitioning)
        currentscape.save(os.path.join(self.output_dir, self.currentscape_filename))
        print("Currentscape saved to " + os.path.join(self.output_dir, self.currentscape_filename))


    def load_preprocessed_data(self):
        print("Loading preprocessed data...")
        pre_dir = os.path.join(self.output_dir, 'preprocessed')
        self.im_path = os.path.join(pre_dir, 'im.csv')
        self.iax_path = os.path.join(pre_dir, 'iax.csv')

        if not os.path.exists(self.im_path) or not os.path.exists(self.iax_path):
            raise FileNotFoundError("Preprocessed data not found. Run full pipeline first.")

        self.taxis = np.load(os.path.join(self.output_dir, 'taxis.npy'))

        time_mask = (self.taxis >= self.tmin) & (self.taxis <= self.tmax)
        selected_times = self.taxis[time_mask]
        selected_cols = [str(int(t)) for t in selected_times]

        usecols_im = ['itype'] + selected_cols
        usecols_iax = ['ref', 'par'] + selected_cols

        # Load only needed columns for membrane and axial currents
        self.im = pd.read_csv(self.im_path, usecols=usecols_im, index_col=0)
        self.iax = pd.read_csv(self.iax_path, usecols=usecols_iax, index_col=[0,1])


    def run_full_pipeline(self):
        """
        Runs the entire data processing pipeline including simulation, preprocessing, calculation, and
        visualization stages. Each step is executed sequentially and is critical for the pipeline
        workflow. The method should be used to execute all stages in the correct order. This method
        does not take any arguments and does not return any value.
        """
        self.run_simulation()
        self.preprocess()
        self.calculate_currentscape()
        self.visualize()
