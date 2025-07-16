# Extended Currentscapes (in vivo demo)

This repository demonstrates an application of the *currentscape* method to **in vivo-like** biophysical simulations of hippocampal CA1 neurons. The analysis visualizes how membrane currents distributed across dendritic compartments contribute to somatic activity during simulated place field traversals.

The core method recursively decomposes axial currents across neuronal compartments to attribute them to underlying membrane currents. The results are rendered as intuitive "currentscapes" — compact plots showing the dynamic contribution of ionic currents to neuronal output.

---

## Repository Elements

- `synaptic_input`: Contains synaptic input files.
- `simulator/`: Builds and simulates the biophysical model neuron.
- `preprocessor/`: Extracts and formats membrane and axial currents from the raw simulation.
- `currentscape_calculator/`: Computes membrane current contributions based on axial current flows.
- `currentscape_visualization/`: Plots currentscapes
- `CurrentscapePipeline.py`: Core pipeline for simulation, preprocessing, analysis, and visualization.

---

## How to Run the Pipeline

Running the full pipeline for a 5-second simulation takes roughly 5-6 hours in total. The simulation step requires about 20 minutes, preprocessing takes 15 minutes, and the currentscape calculation, which is the most time-consuming, takes approximately 1 hour per second of data.

The total runtime depends on the complexity of the model being simulated. However, since each timepoint is processed independently during the currentscape calculation, this step can be easily parallelized to significantly reduce overall processing time.

### 1. **Install Requirements**

Use Python 3.9+ and install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 2. **Compile Mechanism Files**

Before running simulations, you need to compile NEURON's `.mod` files.
Navigate to the folder containing the `.mod` files:

```bash
cd simulator/model/density_mechs
```
And run the `nrnivmodl` tool from NEURON.

If NEURON has trouble finding compiled mechanisms, consider copying them into the root folder and re-running `nrnivmodl`.

More information about NEURON: https://neuron.yale.edu/neuron<br>
More information about working with .mod files: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3263<br>
More information about compiling .mod files: https://nrn.readthedocs.io/en/latest/guide/faq.html#how-do-i-compile-mod-files

---

### 3. **Run Simulation and Currentscape**

You can configure and run the entire pipeline via the `main_full_pipeline.py` script:

```python
from CurrentscapePipeline import CurrentscapePipeline

pipeline = CurrentscapePipeline(
    output_dir='output',
    cluster_seed=0,
    random_seed=30,  # 30–45
    e_input='synaptic_input/Espikes_d10_Ne2000_Re0.5_rseed1_rep0.dat',
    i_input='synaptic_input/Ispikes_d10_Ni200_Ri7.4_rseed1_rep0.dat',
    simulation_time=10.0,  # ms
    target='soma',
    partitioning_strategy='type',
    filename='currentscape_0_5.pdf',
    tmin=0,
    tmax=5
)
pipeline.run_full_pipeline()
```

Alternatively, if you've already run the simulation and preprocessing `from_preprocessed_pipeline.py`:

```python
pipeline = CurrentscapePipeline.from_preprocessed(
    output_dir='output',
    target='soma',
    partitioning_strategy='type',
    filename='currentscape_1_4.pdf',
    tmin=1,
    tmax=4
)
pipeline.load_preprocessed_data()
pipeline.calculate_currentscape()
pipeline.visualize()
```

---

### 4. **Expected Output**

- All outputs are saved in the `output/` folder.
- Key files:
  - `preprocessed/im.csv`, `iax.csv`: Preprocessed membrane and axial currents
  - `results/part_pos_*.csv`, `part_neg_*.csv`: Current contributions
  - `currentscape_*.pdf`: Final currentscape plot.
  - `taxis.npy`: Time vector.
  - `vm.csv`: Table containing the membrane potential arrays of each segment.

The currentscape plot shows:
- The somatic membrane potential
- Total current flowing across the compartment
- The relative contribution of each membrane current to the neuronal activity over time

---

## Interpretation

The generated currentscape enables you to:
- Identify which dendritic regions and current types drive somatic responses

---

## Notes About the Simulations Analyzed in the Article

The simulations discussed in the accompanying article were run with the following parameters:

- `cluster_seed`: 0-15
- `random_seed`: 30–45
- `e_input = 'synaptic_input/Espikes_d10_Ne2000_Re0.5_rseed1_rep0.dat'`: rep0-rep15
- `i_input = 'synaptic_input/Ispikes_d10_Ni200_Ri7.4_rseed1_rep0.dat'`: rep0-rep15

All input files can be found in the `synaptic_input/` directory.