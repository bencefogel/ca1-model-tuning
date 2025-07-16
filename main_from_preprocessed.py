from CurrentscapePipeline import CurrentscapePipeline

output_dir = 'output_5s_simulation'
target = 'soma'
partitioning_strategy = 'type'
tmin = 4800
tmax = 4850
filename = f'currentscape_{target}_{partitioning_strategy}_{tmin}_{tmax}.pdf'

pipeline = CurrentscapePipeline.from_preprocessed(
    output_dir,
    target,
    partitioning_strategy,
    filename,
    tmin,
    tmax
)

pipeline.load_preprocessed_data()
pipeline.calculate_currentscape()
pipeline.visualize()
