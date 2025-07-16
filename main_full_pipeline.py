from CurrentscapePipeline import CurrentscapePipeline


output_dir = 'output_5s_simulation'
cluster_seed = 0
random_seed = 30
e_input = 'synaptic_input/Espikes_d10_Ne2000_Re0.5_rseed1_rep0.dat'
i_input = 'synaptic_input/Ispikes_d10_Ni200_Ri7.4_rseed1_rep0.dat'
simulation_time = 5*1000
target = 'soma'  # can be any dendrite e.g. 'dend5_0'
partitioning_strategy = 'type'  # can be 'type' or 'region'
tmin = 4000
tmax = 4999
filename = f'currentscape_{target}_{partitioning_strategy}_{tmin}_{tmax}.pdf'


pipeline = CurrentscapePipeline(output_dir,
                                cluster_seed,
                                random_seed,
                                e_input,
                                i_input,
                                simulation_time,
                                target,
                                partitioning_strategy,
                                filename,
                                tmin,
                                tmax)
pipeline.run_full_pipeline()
