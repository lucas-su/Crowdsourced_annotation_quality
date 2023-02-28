import os
import pickle
import time
from functools import partial
from multiprocessing import Pool, Process

import krippendorff
import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd

from em import EM
from mcmc import mcmc
def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time

def process_alpha(data):
    alphas = [krippendorff.alpha(reliability_data=data)]
    alphas += [krippendorff.alpha(reliability_data=data.loc[np.eye(data.__len__())[x] != 1]) for x in range(data.__len__())]
    return alphas

def queue_alpha_uerror_metrics(session_dir, session_folder, model):
    with Pool(32) as p:
        for car in car_list:
            for T_dist in T_dist_list:
                for dup in dup_list:
                    for p_fo in p_fo_list:
                        for p_kg in p_kg_list:
                            # for p_kg_u in p_kg_us:

                            result = p.map_async(partial(alpha_uerror_metrics, session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, sessionlen, size), p_kg_u_list)
                            data.loc[(data['session'].values == 'avg') &
                                     (data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == T_dist) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg), 'uerror'] += result.get()
                            if min(data.loc[(data['session'].values == 'avg') &
                                            (data['model'].values == model) &
                                             (data['car'].values == car) &
                                             (data['mode'].values == T_dist) &
                                             (data['dup'].values == dup) &
                                             (data['p_fo'].values == p_fo) &
                                             (data['p_kg'].values == p_kg), 'n_annot_aftr_prun']) == 0:
                                result = p.map(partial(process_krip, session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, nQuestions, size), p_kg_u_list)
                                data.loc[(data['session'].values == 'avg') &
                                         (data['model'].values == model) &
                                         (data['car'].values == car) &
                                         (data['mode'].values == T_dist) &
                                         (data['dup'].values == dup) &
                                         (data['p_fo'].values == p_fo) &
                                         (data['p_kg'].values == p_kg), ['alpha_bfr_prun',
                                                                         'n_annot_aftr_prun',
                                                                         'alpha_aftr_prun',
                                                                         'n_answ_aftr_prun',
                                                                         'pc_aftr_prun',
                                                                         'pc_aftr_prun_total']] = result

def alpha_uerror_metrics(session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, sessionlen, size, p_kg_u):
    with open(f'{session_dir}/{session_folder}/output/{model}_user_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_user = pickle.load(file)
    # error for normal users: modelled T - GT
    u_norm_error = sum(abs(tmp_user.loc[(tmp_user['type']=='normal'), 'T_model']-tmp_user.loc[(tmp_user['type']=='normal'), 'T_given']))

    # error for known good users: modelled T - 1
    u_kg_error = sum((tmp_user.loc[(tmp_user['type']=='KG'), 'T_model']-1)*-1)

    # error for malicious users: modelled T - 1/K
    u_fo_error = sum(abs(tmp_user.loc[(tmp_user['type'] == 'KG'), 'T_model'] - (1/car)))

    return ((u_norm_error + u_kg_error + u_fo_error)/tmp_user.__len__())/sessionlen[model]

def process_krip(session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, nQuestions, size, p_kg_u):
    # do krippendorf pruning and pc calculation, is the same for all sessions so needs to be done only once per condition combination
    with open(f'{session_dir}/{session_folder}/output/{model}_user_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_user = pickle.load(file)
    with open(f'{session_dir}/{session_folder}/output/{model}_annotations_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_annotations = pickle.load(file)
    a = 0
    endindex = -42 if model == 'mcmc' else -12
    users = tmp_user.iloc[:, 4:endindex]
    alpha_bfr_prun = krippendorff.alpha(reliability_data=users)
    while (a < 0.8) & (users.__len__()>2):
        alphas = process_alpha(users)
        a = max(alphas)
        i = alphas.index(max(alphas))
        if i == 0:  # if i == 0, alpha is highest when no annotator is pruned
            break

        users = users.drop(users.iloc[i-1].name)  # -1 because the full set is prepended in a_high

    n_annot_aftr_prun = users.__len__()
    alpha_aftr_prun = a
    n_answ_aftr_prun = np.count_nonzero(np.sum(users.notnull()))

    tmp_annotations['krip'] = [np.nan]*tmp_annotations.__len__()

    for q in range(nQuestions):
        k_w = []
        for k in range(car):
            d_w = 0
            u_include = [x[0] for x in users.iterrows()]
            for d in range(dup):
                if (tmp_annotations.loc[q, f'annot_{d}'] == k) & (tmp_annotations.loc[q, f'id_{d}'] in u_include):
                    d_w += 1
            k_w.append(d_w)
        if sum(k_w) > 0:
            tmp_annotations.loc[q, 'krip'] = k_w.index(max(k_w))

    # determine differences
    diff_k = tmp_annotations.loc[tmp_annotations['krip'].notnull(), 'GT'] - tmp_annotations.loc[tmp_annotations['krip'].notnull(), 'krip']

    # count differences
    diff_k_cnt = (diff_k != 0).sum()

    # proportion correct after pruning and pc corrected for missing items
    pc_aftr_prun =  100 * (1 - (diff_k_cnt / sum(tmp_annotations['krip'].notnull())))
    pc_aftr_prun_total =  pc_aftr_prun*(np.count_nonzero(np.sum(users.notnull()))/nQuestions)
    return alpha_bfr_prun, n_annot_aftr_prun, alpha_aftr_prun, n_answ_aftr_prun, pc_aftr_prun, pc_aftr_prun_total

def makeplaceholderframe(model, idx, datalen, cols):
    assert datalen % 1 == 0
    datalen = int(datalen)
    return pandas.DataFrame(np.concatenate((np.full((1,datalen), idx).T,np.full((1,datalen), model).T, np.zeros((datalen, cols.__len__()-2))), axis=1), columns=cols)

if __name__ == "__main__":
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'

    # car_list = list(range(2, 8))
    # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    # dups = [3,5,7,9]                # duplication factor of the annotators
    # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    # p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]

    car_list = [3]
    T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
    dup_list = [3]
    dup_list = [3]
    p_fo_list = [0.0]
    p_kg_list = [0.0]
    p_kg_u_list = [0.0]

    priors = {'qAlpha': 0.1,
              'aAlpha': 0.1,
              'aBeta': 0.1}

    session_dir = f'sessions/prior-{priors["aAlpha"]}_{priors["aBeta"]}-car{car_list[0]}'
    walk = next(os.walk(session_dir))[1]
    em_sessions = []
    mcmc_sessions = []
    for dir in walk:
        type = next(os.walk(f"{session_dir}/{dir}/output"))[2][0][:2]
        if type == 'em':
            em_sessions.append(dir)
        elif type == 'mc':
            mcmc_sessions.append(dir)
        else:
            raise ValueError

    sessionlen = {'em': em_sessions.__len__(),
                  'mcmc': mcmc_sessions.__len__()}

    iterations = {'em':10,
                  'mcmc': 100}

    # initialize dataset
    for size in ['small']: # ['small', 'medium', 'large']:
        datalen = 2 * car_list.__len__() * T_dist_list.__len__() * dup_list.__len__() * p_fo_list.__len__() * p_kg_list.__len__() * p_kg_u_list.__len__()

        # session denotes the session number, all are needed in memory at once to calculate SD. Session 'avg' is the average over all sessions
        cols = ['session', 'model', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'OBJ', 'pc_m',
                'pc_m_SD', 'pc_n', 'pc_n_SD', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun','n_answ_aftr_prun',
                'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total' ]
        data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
        data.loc[:datalen/2,'model'] = "em"
        data.loc[:datalen / 2, 'iterations'] = iterations['em']
        data.loc[datalen / 2:, 'model'] = "mcmc"
        data.loc[datalen / 2:, 'iterations'] = iterations['mcmc']
        data.loc[:,'session'] = 'avg'

        if size == 'small':
            nQuestions = 10
        elif size == 'medium':
            nQuestions = 200
        else:
            nQuestions = 400

        if em_sessions.__len__()>0:

            # init correct variable values in combined dataframe
            with open(f'{session_dir}/{em_sessions[0]}/output/em_data_size-{size}{"_".join(T_dist_list)}.pickle', 'rb') as file:
                tmp_data = pickle.load(file)
            data.loc[(data['model']=='em')&(data['session']=='avg'),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = np.array(tmp_data.loc[(tmp_data['size']==size),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']])

            # fill frame with EM values
            for em_idx, session in enumerate(em_sessions):
                em_filepath = f'{session_dir}/{session}/output/em_data_size-{size}{"_".join(T_dist_list)}.pickle'
                with open(em_filepath, 'rb') as file:
                    em_data = pickle.load(file)
                data.loc[(data['model']=='em')&(data['session']=='avg'),['pc_m', 'pc_n']] = data.loc[(data['model']=='em')&(data['session']=='avg'),['pc_m', 'pc_n']] + (np.array(em_data.loc[(em_data['size']==size),['pc_m', 'pc_n']]/em_sessions.__len__()))
                data = pandas.concat((data, makeplaceholderframe("em", em_idx, datalen/2, cols)), ignore_index=True)
                data.loc[(data['session']==f'{em_idx}')&(data['model']==f'em'), ['iterations','car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'OBJ', 'pc_m','pc_n']] = np.array(em_data.loc[(em_data['size']==size),['iterations','car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'EM', 'pc_m','pc_n']])
                # queue_alpha_uerror_metrics(session_dir, session, 'em')

        if mcmc_sessions.__len__()>0:

            # init correct variable values in combined dataframe
            with open(f'{session_dir}/{mcmc_sessions[0]}/output/mcmc_data_size-{size}{"_".join(T_dist_list)}.pickle',
                      'rb') as file:
                tmp_data = pickle.load(file)
            data.loc[(data['model'] == 'mcmc') & (data['session'] == 'avg'), ['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = np.array(
                tmp_data.loc[(tmp_data['size'] == size), ['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']])

            # fill frame with MCMC values
            for mc_idx, session in enumerate(mcmc_sessions):
                mcmc_filepath = f'{session_dir}/{session}/output/mcmc_data_size-{size}{"_".join(T_dist_list)}.pickle'
                with open(mcmc_filepath, 'rb') as file:
                    mcmc_data = pickle.load(file)
                data.loc[(data['model'] == 'mcmc')&(data['session']=='avg'), ['pc_m', 'pc_n']] =  data.loc[(data['model'] == 'mcmc')&(data['session']=='avg'), ['pc_m', 'pc_n']] + np.array(mcmc_data.loc[(mcmc_data['size']==size), ['pc_m', 'pc_n']]/mcmc_sessions.__len__())
                data = pandas.concat((data, makeplaceholderframe("mcmc", mc_idx, datalen / 2, cols)), ignore_index=True)
                data.loc[(data['session'] == f'{mc_idx}') & (data['model'] == f'mcmc'), ['iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'OBJ', 'pc_m', 'pc_n']] = np.array(mcmc_data.loc[(mcmc_data['size']==size), ['iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'mcmc', 'pc_m', 'pc_n']])
                # queue_alpha_uerror_metrics(session_dir, session, 'mcmc')

        for T_dist in T_dist_list:
            for model in ['em', 'mcmc']:
                for car in car_list:
                    for T_dist in T_dist_list:
                        for dup in dup_list:
                            for p_fo in p_fo_list:
                                for p_kg in p_kg_list:
                                    for p_kg_u in p_kg_u_list:
                                        # make a slice of all the sessions without the average
                                        dat = data.loc[(data['session'] != 'avg') &
                                                       (data['model'] == model) &
                                                       (data['car'] == car) &
                                                       (data['mode'] == T_dist) &
                                                       (data['dup'] == dup) &
                                                       (data['p_fo'] == p_fo) &
                                                       (data['p_kg'] == p_kg) &
                                                       (data['p_kg_u'] == p_kg_u)]
                                        # determine SD for maj. vote and model
                                        data.loc[(data['session'] == 'avg') &
                                                 (data['model'] == model) &
                                                 (data['car'] == car) &
                                                 (data['mode'] == T_dist) &
                                                 (data['dup'] == dup) &
                                                 (data['p_fo'] == p_fo) &
                                                 (data['p_kg'] == p_kg) &
                                                 (data['p_kg_u'] == p_kg_u), 'pc_n_SD'] = np.std(dat['pc_n'])

                                        data.loc[(data['session'] == 'avg') &
                                                 (data['model'] == model) &
                                                 (data['car'] == car) &
                                                 (data['mode'] == T_dist) &
                                                 (data['dup'] == dup) &
                                                 (data['p_fo'] == p_fo) &
                                                 (data['p_kg'] == p_kg) &
                                                 (data['p_kg_u'] == p_kg_u), 'pc_m_SD'] = np.std(dat['pc_m'])

        with open(f'exports/data_{size}.pickle', 'wb') as file:
            pickle.dump(data, file)