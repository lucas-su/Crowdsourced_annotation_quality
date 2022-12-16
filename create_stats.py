import csv
import os
import pickle

import numpy as np
from tabulate import tabulate
from mcmc import mcmc
from G_EM import EM

def process_stats():
    # error for normal users: modelled T - GT
    u_norm_error = sum(abs(mcmc_user.loc[(mcmc_user['type']=='normal'), 'T_model']-mcmc_user.loc[(mcmc_user['type']=='normal'), 'T_given']))

    # error for known good users: modelled T - 1
    u_kg_error = sum((mcmc_user.loc[(mcmc_user['type']=='KG'), 'T_model']-1)*-1)

    # error for malicious users: modelled T - 1/K
    u_fo_error = sum(abs(mcmc_user.loc[(mcmc_user['type'] == 'KG'), 'T_model'] - (1/car)))

    mcmc_data.loc[(mcmc_data['iterations'].values == iterations['mcmc']) &
                  (mcmc_data['car'].values == car) &
                  (mcmc_data['mode'].values == mode) &
                  (mcmc_data['dup'].values == dup) &
                  (mcmc_data['p_fo'].values == p_fo) &
                  (mcmc_data['p_kg'].values == p_kg) &
                  (mcmc_data['p_kg_u'].values == p_kg_u), 'U_error'] = (u_norm_error + u_kg_error + u_fo_error)/mcmc_user.__len__()

    # u_norm_error = sum(abs(em_user.loc[(em_user['type'] == 'normal'), 'model'] - em_user.loc[(em_user['type'] == 'normal'), 'GT']))
    # u_kg_error = sum(abs(em_user.loc[(em_user['type'] == 'KG'), 'model'] - 1))
    # u_fo_error = sum(abs(em_user.loc[(em_user['type'] == 'KG'), 'model'] - (1 / car)))
    #
    # em_data.loc[(em_data['iterations'].values == iterations['EM']) &
    #               (em_data['car'].values == car) &
    #               (em_data['mode'].values == mode) &
    #               (em_data['dup'].values == dup) &
    #               (em_data['p_fo'].values == p_fo) &
    #               (em_data['p_kg'].values == p_kg) &
    #               (em_data['p_kg_u'].values == p_kg_u), 'U_error'] = (u_norm_error + u_kg_error + u_fo_error) / em_user.__len__()

def export():
    slice = mcmc_data.loc[(mcmc_data['mode'].values == modes[0]) & (mcmc_data['car'].values == 2), ['car', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'U_error', 'pc_m', 'pc_n']]
    slice.to_csv(f'exports/mcmc_{mode}.csv', index=False,header=False)


def c_table():
    slice = mcmc_data.loc[(mcmc_data['mode'].values == modes[0]) & (mcmc_data['car'].values == 2),
                          ['car', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'U_error', 'pc_m', 'pc_n']]


    # if colnameloc == 'top':
    #     ncolumns = 4 + 5
    table = "\\begin{tabular}{lr}\n"
    table += "\\hline"
    table += "\\hline"
    print(table)

if __name__ == "__main__":
    session_folder = "session_2022-12-13_11-33-52"
    os.makedirs(f'{os.getcwd()}/exports', exist_ok=True)
    iterations = {'em':10,
                  'mcmc': 40}

    # car_list = list(range(2,8))     # cardinality of the questions
    car_list = [2]
    modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']

    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]


    with open(f'data/{session_folder}/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
        mcmc_data = pickle.load(file)
    # with open(f'data/{session_folder}/em_data_{"_".join(modes)}.pickle', 'rb') as file:
    #     em_data = pickle.load(file)

    # placeholder for new column
    mcmc_data.loc[:,'U_error'] = np.zeros(mcmc_data.__len__())
    for car in car_list:
        for mode in modes[:1]:
            for dup in dups:
                for p_fo in p_fos:
                    for p_kg in p_kgs:
                        for p_kg_u in p_kg_us:
                            # open EM files
                            # with open(f'data/{session_folder}/em_annotations_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["em"]}.pickle', 'rb') as file:
                            #     em_annotations = pickle.load(file)
                            # with open(f'data/{session_folder}/em_user_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["em"]}.pickle', 'rb') as file:
                            #     em_user = pickle.load(file)
                            # open MCMC files
                            with open(f'data/{session_folder}/mcmc_annotations_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                mcmc_annotations = pickle.load(file)
                            with open(f'data/{session_folder}/mcmc_user_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                mcmc_user = pickle.load(file)
                            process_stats()
    export()
