import pickle

import krippendorff
import matplotlib.pyplot as plt
import pandas
import numpy as np
from G_EM import EM
from mcmc import mcmc

def process_alpha(data):
    alphas = [krippendorff.alpha(reliability_data=data)]
    alphas += [krippendorff.alpha(reliability_data=data.loc[np.eye(data.__len__())[x] != 1]) for x in range(data.__len__())]
    return alphas

def alpha_uerror_metrics(session_folder, model):

    for car in car_list:
        for mode in modes[:1]:
            for dup in dups:
                for p_fo in p_fos:
                    for p_kg in p_kgs:
                        for p_kg_u in p_kg_us:
                            with open(f'data/{session_folder}/{model}_annotations_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
                                tmp_annotations = pickle.load(file)
                            with open(f'data/{session_folder}/{model}_user_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
                                tmp_user = pickle.load(file)
                            # error for normal users: modelled T - GT
                            u_norm_error = sum(abs(tmp_user.loc[(tmp_user['type']=='normal'), 'T_model']-tmp_user.loc[(tmp_user['type']=='normal'), 'T_given']))

                            # error for known good users: modelled T - 1
                            u_kg_error = sum((tmp_user.loc[(tmp_user['type']=='KG'), 'T_model']-1)*-1)

                            # error for malicious users: modelled T - 1/K
                            u_fo_error = sum(abs(tmp_user.loc[(tmp_user['type'] == 'KG'), 'T_model'] - (1/car)))

                            data.loc[(data['model'].values == model) &
                                (data['car'].values == car) &
                                 (data['mode'].values == mode) &
                                 (data['dup'].values == dup) &
                                 (data['p_fo'].values == p_fo) &
                                 (data['p_kg'].values == p_kg) &
                                 (data['p_kg_u'].values == p_kg_u), 'uerror'] += ((u_norm_error + u_kg_error + u_fo_error)/tmp_user.__len__())/sessionlen[model]

                            # do krippendorf pruning and pc calculation, is the same for all sessions so needs to be done only once
                            if data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'n_annot_aftr_prun'].item() != 0:
                                continue
                            a = 0
                            endindex = -42 if model == 'mcmc' else -12
                            users = tmp_user.iloc[:, 4:endindex]
                            while (a < 0.8) & (users.__len__()>2):
                                alphas = process_alpha(users)
                                a = max(alphas)
                                i = alphas.index(max(alphas))
                                if i == 0:  # if i == 0, alpha is highest when no annotator is pruned
                                    break

                                users = users.drop(users.iloc[i-1].name)  # -1 because the full set is prepended in a_high

                            data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'n_annot_aftr_prun'] = users.__len__()
                            data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'alpha_aftr_prun'] = a
                            data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'n_answ_aftr_prun'] = np.count_nonzero(np.sum(users.notnull()))

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


                            data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'pc_aftr_prun'] =  100 * (1 - (diff_k_cnt / sum(tmp_annotations['krip'].notnull())))
                            data.loc[(data['model'].values == model) &
                                     (data['car'].values == car) &
                                     (data['mode'].values == mode) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg) &
                                     (data['p_kg_u'].values == p_kg_u), 'pc_aftr_prun_total'] =  data.loc[(data['model'].values == model) &
                                                     (data['car'].values == car) &
                                                     (data['mode'].values == mode) &
                                                     (data['dup'].values == dup) &
                                                     (data['p_fo'].values == p_fo) &
                                                     (data['p_kg'].values == p_kg) &
                                                     (data['p_kg_u'].values == p_kg_u), 'pc_aftr_prun']*(np.count_nonzero(np.sum(users.notnull()))/nQuestions)


if __name__ == "__main__":
    # model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'
    em_sessions = ["session_2022-12-16_11-34-04"]
    mcmc_sessions = ["session_2022-12-13_11-33-52"]
    nQuestions = 200
    sessionlen = {'em': em_sessions.__len__(),
                  'mcmc': mcmc_sessions.__len__()}

    iterations = {'em':10,
                  'mcmc': 40}
    # car_list = list(range(2, 8))
    car_list = list(range(2, 6))
    modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]

    datalen = 2*car_list.__len__()*modes.__len__()*dups.__len__()*p_fos.__len__()*p_kgs.__len__()*p_kg_us.__len__()
    cols = ['model', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'EM', 'pc_m', 'pc_n', 'uerror', 'n_annot_aftr_prun','n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total' ]
    data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
    data.loc[:datalen/2,'model'] = "em"
    data.loc[:datalen / 2, 'iterations'] = 10
    data.loc[datalen / 2:, 'model'] = "mcmc"
    data.loc[datalen / 2:, 'iterations'] = 40

    # init correct values in combined dataframe
    with open(f'data/{em_sessions[0]}/em_data_{"_".join(modes)}.pickle', 'rb') as file:
        tmp_data = pickle.load(file)
    data.loc[(data['model']=='em'),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = tmp_data.loc[:,['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']]

    with open(f'data/{mcmc_sessions[0]}/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
        tmp_data = pickle.load(file)
    data.loc[(data['model']=='mcmc'),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = tmp_data.loc[:,['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']]

    for session in em_sessions:
        em_filepath = f'data/{session}/em_data_{"_".join(modes)}.pickle'
        with open(em_filepath, 'rb') as file:
            em_data = pickle.load(file)
        data.loc[(data['model']=='em'),['pc_m', 'pc_n']] += em_data.loc[:,['pc_m', 'pc_n']]/em_sessions.__len__()
        alpha_uerror_metrics(session, 'em')


    for session in mcmc_sessions:
        mcmc_filepath = f'data/{session}/mcmc_data_{"_".join(modes)}.pickle'
        with open(mcmc_filepath, 'rb') as file:
            mcmc_data = pickle.load(file)
        data.loc[(data['model'] == 'mcmc'), ['pc_m', 'pc_n']] += mcmc_data.loc[:, ['pc_m', 'pc_n']]/em_sessions.__len__()
        alpha_uerror_metrics(session, 'mcmc')


    # inits
    p_kg = p_kgs[0]
    mode = modes[0]