import multiprocessing
import numpy as np
import platform 
car_list = [3,5,7]

# T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]    
# T_dist_list = [f'beta{round(flt*18+1, 2)}_{round(20-(flt*18+1), 2)}' for flt in np.arange(0, 1.1, 0.1)]
T_dist_list = [f'T_else{round(flt, 2)}' for flt in np.arange(0.0, 1.1, 0.1)]
ncpu = multiprocessing.cpu_count()
debug = False

dup_list = [5]
p_fo_list = [0.0]
p_kg_list = [0.0]
p_kg_u_list = [0.0,0.05]
if debug:
    datasetsize_list = ['debug'] 
else:
    datasetsize_list = ['medium', 'large'] #['small','medium','large']
# datasetsize = datasetsize_list[0]

# sampling parameters

    # priors should always be a float

            # 'aAlpha':15.,
            # 'aBeta':0.15}

def set_nQuestions(datasetsize):
    if datasetsize == 'small': # avg 3 annots per annotator
        nQuestions = 3.
    elif datasetsize == 'medium': # avg 6 annots per annotator
        nQuestions = 6.
    elif datasetsize == 'large': # avg 15 annots per annotator
        nQuestions = 15.
    else:
        raise(ValueError,'Datasetsize should be "small", "medium", or "large"')
    return nQuestions
def set_priors(nQuestions, car):
    # average number of annotations per annotator can(?) determine alpha and beta prior. (nQuestions-1, (nQuestions-1)/100) seems to work well
    priors = {'qAlpha': .001}  # should be dependent on dup: probably dup/10

    a = 10*car
    b = 1
    fraction = 5

    priors['aAlpha'] = round((nQuestions*(a/(a+b)))/fraction, 1)
    priors['aBeta'] =  round((nQuestions*(b/(a+b)))/fraction, 1)

    for pr in priors.values():
        assert type(pr) == float, 'Priors need to be floats'
    return priors

# decrease annotators for quick debugging
if debug:
    nAnnotator = 5
else:
    nAnnotator = 20

if platform.system() == 'Windows': # running local: fewer demands
    warmup = 25
    nSamples = 3
    sample_interval = 1
    keep_samples_list = [3]
    nModels = 5
else:
    warmup = 20
    nSamples = 3
    # keep a sample every sample_interval iterations
    sample_interval = 1
    # n samples to keep
    keep_samples_list = [5]
    nModels = 15
    T_dist_list = [f'T_else{round(flt, 2)}' for flt in np.arange(0.0, 1.1, 0.1)]

# create data settings
c_data_mal_T = True # assumes 'malicious' users at T=0 if true, meaning that they give anything but the correct answers. False means uniform chance over all answers at T=0