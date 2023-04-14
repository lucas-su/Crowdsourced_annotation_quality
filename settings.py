import multiprocessing
from datetime import datetime

import numpy as np
import platform 
car_list = [2,4]

# T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
beta_min = 0.01
beta_max = 0.8
sweeps = {'beta_small':[f'beta2{round(flt, 2)}_{round(beta_max-flt, 2)}' for flt in np.linspace(beta_min, beta_max-beta_min, 11)]}
          # "propT": [f'propT_{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]}

ncpu = multiprocessing.cpu_count()
debug = False

dup_list = [9]
p_fo_list = [0.0]
kg_q_list = [0,5]
kg_u_list = [0,1]


if debug:
    datasetsize_list = ['debug'] 
else:
    datasetsize_list = ['medium', 'large']# ['small','medium','large', 'xlarge']

# decrease annotators for quick debugging
if debug:
    nAnnotator = 3
    dup_list = [2]
else:
    nAnnotator = 20

if debug:
    warmup = 1
    nSamples = 2
    sample_interval = 2
    keep_samples_list = [2]
    nModels = 2

elif platform.system() == 'Windows': # running local: fewer demands
    warmup = 10
    nSamples = 1
    sample_interval = 1
    keep_samples_list = [1]
    nModels = 5
else:
    warmup = 15
    nSamples = 3 # number of samples per iteration
    sample_interval = 1 # keep a sample every sample_interval iterations
    keep_samples_list = [5] # n samples to keep
    nModels = 5


# create data settings
c_data_mal_T = True # assumes 'malicious' users at T=0 if true, meaning that they give anything but the correct answers. False means uniform chance over all answers at T=0

def set_session_dir(size, sweeptype, car, dup, p_fo, kg_q, kg_u):

    session_dir = f'sessions/datasetsize_{size}/sweeptype_{sweeptype}/cardinality_{car}/dup_{dup}/p_fo_{p_fo}/kg_q_{kg_q}/kg_u_{kg_u}/'
    return session_dir

def set_priors():

    priors = {'qAlpha': 1e-5}

    a = 1.5
    b = 1.
    # fraction =2

    priors['aAlpha'] = a
    priors['aBeta'] =  b

    for pr in priors.values():
        assert type(pr) == float, 'Priors need to be floats'
    print(f'Priors set to: {priors}')
    return priors