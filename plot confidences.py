import pickle
from collections import Counter

import numpy as np
import pandas
from matplotlib import pyplot as plt


def plot():
    posteriorindices = (10*[False])+[x%3==0 for x in range(30)]
    uncertainty = pandas.Series(0, index=np.arange(posteriorindices.__len__()))
    for q in range(annotations.__len__()):
        cnt = Counter(annotations.loc[q, [f'model_{i}' for i, x in enumerate(posteriorindices) if x]])
        # uncertainty per label: 1-((|Lm|*car)/(|m|*car))
        # uncertainty[q] = (1-((cnt.most_common(1)[0][1])/(sum(list(cnt.values())))))/car
        mc = cnt.most_common(2)
        if mc.__len__()>1:
            uncertainty[q] = 1-np.prod([mc[0][1]/(mc[0][1]+mc[i+1][1]) for i in range(mc.__len__()-1)]) # product of all probs wrt all other options
        else:
            uncertainty[q] = 0
    uncertainty.plot.bar()
    plt.show()



if __name__ == "__main__":
    model = "mcmc"  # options "em" or "mcmc"
    session_folder = 'session_2022-12-01_15-54-47'
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\{model} confidence plots\\'

    # car_list = list(range(2, 8))
    car_list = list(range(2, 5))

    iterations_list = [5, 20]
    modes = ['uniform', 'gaussian', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # inits
    # p_kg = p_kgs[0]
    p_kg = 0.1
    # mode = modes[0]
    T_dist = 'beta4_2'
    # iterations = iterations_list[0]
    iterations = 40
    # dup = dups[0]
    dup = 9
    # car = car_list[0]
    car = 5
    # p_fo = p_fos[0]
    p_fo = 0.1

    filepath = f'data/{session_folder}/mcmc_annotations_p_kg-{p_kg}_data_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_iters-{iterations}.pickle',
    # filepath = 'data/mcmc_data_uniform_gaussian_gaussian50_50_single0_single1_beta1_3_beta3_1.pickle'

    with open(f'data/{session_folder}/mcmc_annotations_p_kg-{p_kg}_data_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_iters-{iterations}.pickle', 'rb') as file:
        annotations = pickle.load(file)

    plot()