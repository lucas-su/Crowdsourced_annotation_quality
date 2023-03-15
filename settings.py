import multiprocessing
from datetime import datetime

import numpy as np
import platform 
car_list = [2, 3, 4]

# T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]    
# T_dist_list = [f'beta{round(flt*18+1, 2)}_{round(20-(flt*18+1), 2)}' for flt in np.arange(0, 1.1, 0.1)]
T_dist_list = [f'T_else{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
ncpu = multiprocessing.cpu_count()
debug = False

dup_list = [5]
p_fo_list = [0.0]
p_kg_list = [0.]
p_kg_u_list = [0.0]

if debug:
    datasetsize_list = ['debug'] 
else:
    datasetsize_list = ['medium']#, 'large'] #['small','medium','large']

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
    warmup = 20
    nSamples = 3
    sample_interval = 1
    keep_samples_list = [1]
    nModels = 5
else:
    warmup = 25
    nSamples = 3 # number of samples per iteration
    sample_interval = 1 # keep a sample every sample_interval iterations
    keep_samples_list = [5] # n samples to keep
    nModels = 5


# create data settings
c_data_mal_T = True # assumes 'malicious' users at T=0 if true, meaning that they give anything but the correct answers. False means uniform chance over all answers at T=0

def set_session_dir(size, car, dup, p_fo, p_kg, p_kg_u):

    session_dir = f'sessions/datasetsize_{size}/cardinality_{car}/dup_{dup}/p_fo_{p_fo}/p_kg_{p_kg}/p_kg_u_{p_kg_u}/'
    return session_dir


def set_nQuestions(datasetsize):
    if datasetsize == 'debug':
        nQuestions = 2.
    elif datasetsize == 'small': # avg 3 annots per annotator
        nQuestions = 3.
    elif datasetsize == 'medium': # avg 6 annots per annotator
        nQuestions = 6.
    elif datasetsize == 'large': # avg 15 annots per annotator
        nQuestions = 15.
    else:
        raise(ValueError,'Datasetsize should be "small", "medium", or "large"')
    return nQuestions
def set_priors(nQuestions, car, dup):

    priors = {'qAlpha': round(dup/10, 3)}

    a = 2
    b = 1
    fraction = 5 # gwenn advised 5

    priors['aAlpha'] = round((nQuestions*(a/(a+b)))/fraction, 3)
    priors['aBeta'] =  round((nQuestions*(b/(a+b)))/fraction, 3)

    for pr in priors.values():
        assert type(pr) == float, 'Priors need to be floats'
    print(f'Priors set to: {priors}')
    return priors