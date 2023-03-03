import multiprocessing
import numpy as np
import platform 
car_list = [5]
T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
dup_list = [5]
p_fo_list = [0.0, 0.1]
p_kg_list = [0.0, 0.1]
p_kg_u_list = [0.0, 0.1]
datasetsize_list = ['large'] #['small','medium','large']
datasetsize = datasetsize_list[0]

# priors should always be a float
priors = {'qAlpha':1.,
            'aAlpha':1.,
            'aBeta':0.5}
nAnnot = 30
nModels = 3

ncpu = multiprocessing.cpu_count()

if platform.system() == 'Windows': # for quick debug
    warmup = 3
    nSamples = 1
    sample_interval = 1
    keep_samples_list = [5]
else:
    warmup = 10
    nSamples = 5
    # keep a sample every sample_interval iterations
    sample_interval = 2
    # n samples to keep
    keep_samples_list = [10]


