import time
from settings import *
import krippendorff

from em import EM
from mcmc import *

def process_alpha(data):
    alphas = [krippendorff.alpha(reliability_data=data)]
    alphas += [krippendorff.alpha(reliability_data=data.loc[np.eye(data.__len__())[x] != 1]) for x in range(data.__len__())]
    return alphas

def queue_alpha_uerror_metrics(data, session_dir, session_folder, model, iterations, sessionlen, nQuestions, size):
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
                                     (data['T_dist'].values == T_dist) &
                                     (data['dup'].values == dup) &
                                     (data['p_fo'].values == p_fo) &
                                     (data['p_kg'].values == p_kg), 'uerror'] += result.get()
                            if min(data.loc[(data['session'].values == 'avg') &
                                            (data['model'].values == model) &
                                             (data['car'].values == car) &
                                             (data['T_dist'].values == T_dist) &
                                             (data['dup'].values == dup) &
                                             (data['p_fo'].values == p_fo) &
                                             (data['p_kg'].values == p_kg), 'n_annot_aftr_prun']) == 0:
                                result = p.map(partial(process_krip, session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, nQuestions, size), p_kg_u_list)
                                data.loc[(data['session'].values == 'avg') &
                                         (data['model'].values == model) &
                                         (data['car'].values == car) &
                                         (data['T_dist'].values == T_dist) &
                                         (data['dup'].values == dup) &
                                         (data['p_fo'].values == p_fo) &
                                         (data['p_kg'].values == p_kg), ['alpha_bfr_prun',
                                                                         'n_annot_aftr_prun',
                                                                         'alpha_aftr_prun',
                                                                         'n_answ_aftr_prun',
                                                                         'pc_aftr_prun',
                                                                         'pc_aftr_prun_total']] = result

def alpha_uerror_metrics(session_dir, session_folder, model, car, T_dist, dup, p_fo, p_kg, iterations, sessionlen, size, p_kg_u):
    with open(f'{session_dir}/{session_folder}/output/{model}_user_data_size-{size}_T_dist-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations[model]}.pickle', 'rb') as file:
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


def process_model(model, session_dir, sessions, data, cols, size, car, dup, p_fo, p_kg, p_kg_u):
    with open(f'{session_dir}/{sessions[0]}/output/{model}_data_{"_".join(T_dist_list)}.pickle', 'rb') as file:
        tmp_data = pickle.load(file)
    data.loc[(data['model'] == model) & (data['session'] == 'avg'),
    ['car', 'T_dist', 'dup', 'p_fo', 'p_kg','p_kg_u']] = np.array(tmp_data.loc[(tmp_data['size'] == size) & (tmp_data['car'] == car) & (tmp_data['p_kg_u'] == p_kg_u),
    ['car', 'T_dist','dup', 'p_fo','p_kg', 'p_kg_u']])

    # fill frame with values
    for idx, session in enumerate(sessions):
        filepath = f'{session_dir}/{session}/output/{model}_data_{"_".join(T_dist_list)}.pickle'
        with open(filepath, 'rb') as file:
            tmp_data = pickle.load(file)
        data.loc[(data['model'] == model) & (data['session'] == 'avg'),
            ['pc_m', 'pc_n', 'CertaintyQ', 'CertaintyA']] = data.loc[(data['model'] == model) & (data['session'] == 'avg'),
            ['pc_m', 'pc_n', 'CertaintyQ', 'CertaintyA']] + np.array(tmp_data.loc[:, ['pc_m', 'pc_n', 'CertaintyQ', 'CertaintyA']] / sessions.__len__())
        data = pandas.concat((data, makeplaceholderframe(model, idx, T_dist_list.__len__(), cols)), ignore_index=True)
        data.loc[(data['session'] == f'{idx}') & (data['model'] == f'{model}'),
        ['iterations', 'car', 'T_dist', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'OBJ', 'pc_m', 'pc_n']] = np.array(tmp_data.loc[:,
        ['iterations', 'car', 'T_dist', 'dup', 'p_fo', 'p_kg', 'p_kg_u', f'{model}', 'pc_m', 'pc_n']])
        # queue_alpha_uerror_metrics(session_dir, session, model,iterations, sessionlen)

        for T_dist in T_dist_list:
            # make a slice of all the sessions without the average
            dat = data.loc[(data['session'] != 'avg') &
                           (data['model'] == model) &
                           (data['car'] == car) &
                           (data['T_dist'] == T_dist) &
                           (data['dup'] == dup) &
                           (data['p_fo'] == p_fo) &
                           (data['p_kg'] == p_kg) &
                           (data['p_kg_u'] == p_kg_u)]
            # determine SD for maj. vote and model
            data.loc[(data['session'] == 'avg') &
                     (data['model'] == model) &
                     (data['car'] == car) &
                     (data['T_dist'] == T_dist) &
                     (data['dup'] == dup) &
                     (data['p_fo'] == p_fo) &
                     (data['p_kg'] == p_kg) &
                     (data['p_kg_u'] == p_kg_u), 'pc_n_SD'] = np.std(dat['pc_n'])

            data.loc[(data['session'] == 'avg') &
                     (data['model'] == model) &
                     (data['car'] == car) &
                     (data['T_dist'] == T_dist) &
                     (data['dup'] == dup) &
                     (data['p_fo'] == p_fo) &
                     (data['p_kg'] == p_kg) &
                     (data['p_kg_u'] == p_kg_u), 'pc_m_SD'] = np.std(dat['pc_m'])
    return data

def main(size, car, dup, p_fo, p_kg, p_kg_u):
    # latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'

    session_dir = set_session_dir(size, car, dup, p_fo, p_kg, p_kg_u)
    try:
        walk = next(os.walk(session_dir))[1]
    except:
        pass
    em_sessions = []
    mcmc_sessions = []
    for dir in walk:
        try:
            type = next(os.walk(f"{session_dir}/{dir}/output"))[2][0][:2]
        except:
            pass
        if type == 'em':
            em_sessions.append(dir)
        elif type == 'mc':
            mcmc_sessions.append(dir)
        else:
            raise ValueError

    # initialize dataset
    datalen = 2* T_dist_list.__len__()

    # session denotes the session number, all are needed in memory at once to calculate SD. Session 'avg' is the average over all sessions
    cols = ['session', 'model', 'car', 'T_dist', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'OBJ', 'pc_m', 'CertaintyQ',
            'CertaintyA', 'pc_m_SD', 'pc_n', 'pc_n_SD', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun','n_answ_aftr_prun',
            'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total' ]
    data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
    data.loc[:datalen/2,'model'] = "em"
    data.loc[datalen / 2:, 'model'] = "mcmc"
    data.loc[:,'session'] = 'avg'

    if em_sessions.__len__()>0:
        data = process_model('em', session_dir, em_sessions, data, cols, size, car, dup, p_fo, p_kg, p_kg_u)


    if mcmc_sessions.__len__()>0:
        data = process_model('mcmc', session_dir, mcmc_sessions, data, cols, size, car, dup, p_fo, p_kg, p_kg_u)

    with open(f'{session_dir}/stats.pickle', 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    for dup in dup_list:
        for p_fo in p_fo_list:
            for p_kg in p_kg_list:
                for p_kg_u in p_kg_u_list:
                    for size in datasetsize_list:
                        for car in car_list:
                            main(size, car, dup, p_fo, p_kg, p_kg_u)