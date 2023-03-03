import numpy as np
car_list = [5]
T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
dup_list = [5]
p_fo_list = [0.0, 0.1]
p_kg_list = [0.0, 0.1]
p_kg_u_list = [0.0, 0.1]
datasetsize_list = ['large'] #['small','medium','large']
datasetsize = datasetsize_list[0]
priors = {'qAlpha':1,
            'aAlpha':1,
            'aBeta':0.5}
nAnnot = 30
nModels = 10