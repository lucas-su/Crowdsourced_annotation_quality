import pickle

import numpy as np
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import pandas as pd

from mcmc import mcmc
import krippendorff

def a_high(data):
    alphas = [krippendorff.alpha(reliability_data=data)]
    alphas += [krippendorff.alpha(reliability_data=data.loc[np.eye(data.__len__())[x] != 1]) for x in range(data.__len__())]
    return alphas

if __name__ == "__main__":
    em_sessions = ["session_2022-12-16_11-34-04"]
    mcmc_sessions = ["session_2022-12-13_11-33-52"]
    modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']

    iterations = {'em':10,
                  'mcmc': 40}
    car_list = list(range(2, 6))
    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]
    for session in mcmc_sessions:
        mcmc_filepath = f'data/{session}/mcmc_data_{"_".join(modes)}.pickle'
        with open(mcmc_filepath, 'rb') as file:
            mcmc_data = pickle.load(file)
        for car in car_list:
            for mode in modes[:1]:
                for dup in dups:
                    for p_fo in p_fos:
                        for p_kg in p_kgs:
                            for p_kg_u in p_kg_us:
                                with open(f'data/{session}/mcmc_annotations_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                    mcmc_annotations = pickle.load(file)
                                with open(f'data/{session}/mcmc_user_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                    mcmc_user = pickle.load(file)
                                q = mcmc_user.iloc[:,4:-42]
                                a = krippendorff.alpha(reliability_data=q)
                                while a < 0.8:
                                    alphas = a_high(q)
                                    a = max(alphas)
                                    i = alphas.index(max(alphas))
                                    if i == 0: # if i == 0, alpha is highest when no annotator is pruned
                                        break
                                    q = q.drop(i-1).reset_index(drop=True) # -1 because the full set is prepended in a_high
