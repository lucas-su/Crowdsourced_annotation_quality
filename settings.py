import multiprocessing
import numpy as np
import platform 
car_list = [5]

# T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]    
# T_dist_list = [f'beta{round(flt*18+1, 2)}_{round(20-(flt*18+1), 2)}' for flt in np.arange(0, 1.1, 0.1)]
T_dist_list = [f'T_else{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]    

debug = False

dup_list = [5]
p_fo_list = [0.0]
p_kg_list = [0.0]
p_kg_u_list = [0.0]
if debug:
    datasetsize_list = ['debug'] 
else:
    datasetsize_list = ['large'] #['small','medium','large']
datasetsize = datasetsize_list[0]

# priors should always be a float
priors = {'qAlpha':0.1,
            'aAlpha':0.7,
            'aBeta':0.1}

if debug:
    nAnnot = 5
else:
    nAnnot = 20

nModels = 4

ncpu = multiprocessing.cpu_count()

if platform.system() == 'Windows': # for quick debug
    warmup = 15
    nSamples = 3
    sample_interval = 1
    keep_samples_list = [3]
else:
    warmup = 15
    nSamples = 3
    # keep a sample every sample_interval iterations
    sample_interval = 2
    # n samples to keep
    keep_samples_list = [5]

