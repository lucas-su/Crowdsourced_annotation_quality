import multiprocessing
import numpy as np
import platform 
car_list = [3]

# T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]    
# T_dist_list = [f'beta{round(flt*18+1, 2)}_{round(20-(flt*18+1), 2)}' for flt in np.arange(0, 1.1, 0.1)]
T_dist_list = [f'T_else{round(flt, 2)}' for flt in np.arange(0.0, 1.1, 0.1)]
ncpu = multiprocessing.cpu_count()
debug = False

dup_list = [5]
p_fo_list = [0.0]
p_kg_list = [0.0]
p_kg_u_list = [0.0]
if debug:
    datasetsize_list = ['debug'] 
else:
    datasetsize_list = ['medium'] #['small','medium','large']
datasetsize = datasetsize_list[0]

# sampling parameters

    # priors should always be a float
priors = {'qAlpha':.1}
            # 'aAlpha':15.,
            # 'aBeta':0.15}

# average number of annotations per annotator can(?) determine alpha and beta prior. (nQuestions-1, (nQuestions-1)/100) seems to work well
if datasetsize == 'small':
    priors['aAlpha'] = 3.
    priors['aBeta'] = .002
elif datasetsize == 'medium':
    priors['aAlpha'] = 5.
    priors['aBeta'] = .05
elif datasetsize == 'large':
    priors['aAlpha'] = 5.
    priors['aBeta'] = .1
else:
    raise ValueError

for pr in priors.values():
    assert type(pr) == float


# decrease annotators for quick debugging
if debug:
    nAnnot = 5
else:
    nAnnot = 20

nModels = 4


if platform.system() == 'Windows': # running local: fewer demands
    warmup = 20
    nSamples = 3
    sample_interval = 1
    keep_samples_list = [3]
else:
    warmup = 20
    nSamples = 3
    # keep a sample every sample_interval iterations
    sample_interval = 1
    # n samples to keep
    keep_samples_list = [5]

# create data settings
c_data_mal_T = True # assumes 'malicious' users at T=0 if true, meaning that they give anything but the correct answers. False means uniform chance over all answers at T=0