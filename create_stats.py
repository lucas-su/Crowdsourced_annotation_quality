import re
import numpy as np, os, platform, pandas, pickle
import krippendorff
from scipy.stats import beta
from multiprocessing import Pool
from functools import partial
from scipy.stats import ttest_ind

from em import EM
from mcmc import mcmc

def process_alpha(data):
    try:
        with np.errstate(all='raise'):
            OK_permutations = []
            for x in range(data.__len__()):
                # check if there is at least one overlapping question after removing annotator
                if sum(np.all(~np.isnan(data.loc[np.eye(data.__len__())[x] != 1]).T, axis=1))>0:
                    # check if there is a disagreement among the overlapping questions: krippendorff alpha is undefined if all overlapping answers agree

                    if np.all(np.diff(np.array(data.loc[np.eye(data.__len__())[x] != 1,np.all(~np.isnan(data.loc[np.eye(data.__len__())[x] != 1]), axis=0)]), axis=0)==0):
                        OK_permutations.append(x)

            alphas = [krippendorff.alpha(reliability_data=data)]

            alphas += [krippendorff.alpha(reliability_data=data.loc[np.eye(data.__len__())[x] != 1]) if x in OK_permutations else -1 for x in range(data.__len__())]
    except Exception as e:
        print(f'Process alpha error {e}')
    return alphas

def queue_alpha_uerror_metrics(model, session_dir, session, session_idx, data, T_dist_list, car, dup, p_fo, kg_q, kg_u):

    iterations = {'mcmc':5, 'em':100}
    with Pool(32) as p:
        result = p.map(partial(alpha_uerror_metrics, session_dir, model, car, iterations, session), T_dist_list)
        for T_dist, res in result:
            data.loc[(data['session'].values == f'{session_idx}') &
                     (data['model'].values == model) &
                     (data['car'].values == car) &
                     (data['T_dist'].values == T_dist) &
                     (data['dup'].values == dup) &
                     (data['p_fo'].values == p_fo) &
                     (data['kg_q'].values == kg_q) &
                     (data['kg_u'].values == kg_u), 'uerror'] += res

        result = p.map(partial(process_krip, session_dir, model, car, dup, iterations, session), T_dist_list)
        for T_dist, res in result:
            data.loc[(data['session'].values == f'{session_idx}') &
                     (data['model'].values == model) &
                     (data['car'].values == car) &
                     (data['T_dist'].values == T_dist) &
                     (data['dup'].values == dup) &
                     (data['p_fo'].values == p_fo) &
                     (data['kg_q'].values == kg_q)&
                     (data['kg_u'].values == kg_u), ['alpha_bfr_prun',
                                                     'n_annot_aftr_prun',
                                                     'alpha_aftr_prun',
                                                     'n_answ_aftr_prun',
                                                     'pc_aftr_prun',
                                                     'pc_krip']] = res
        # print('q_alpha_done')

def alpha_uerror_metrics(session_dir, model, car, iterations, session, T_dist):
    with open(f'{session_dir}/{session}/output/{model}_user_T_dist-{T_dist}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_user = pickle.load(file)
    # error for normal users: modelled T - GT
    u_norm_error = sum(abs(tmp_user.loc[(tmp_user['type']=='normal'), 'T_model']-tmp_user.loc[(tmp_user['type']=='normal'), 'T_given']))

    # error for known good users: modelled T - 1
    u_kg_error = sum((tmp_user.loc[(tmp_user['type']=='KG'), 'T_model']-1)*-1)

    # error for malicious users: modelled T - 1/K
    u_fo_error = sum(abs(tmp_user.loc[(tmp_user['type'] == 'KG'), 'T_model'] - (1/car)))

    return T_dist, ((u_norm_error + u_kg_error + u_fo_error)/tmp_user.__len__())

def process_krip(session_dir, model, car, dup, iterations, session, T_dist):
    # do krippendorf pruning and pc calculation
    with open(f'{session_dir}/{session}/output/{model}_user_T_dist-{T_dist}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_user = pickle.load(file)
    with open(f'{session_dir}/{session}/output/{model}_annotations_T_dist-{T_dist}_iters-{iterations[model]}.pickle', 'rb') as file:
        tmp_annotations = pickle.load(file)
    a = 0

    users = tmp_user.iloc[:, 4:]
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

    for q in range(tmp_annotations.__len__()):
        k_weight = np.zeros(car)
        for k in range(car):
            # d_w = 0
            u_include = [x[0] for x in users.iterrows()]
            for d in range(dup):
                if (tmp_annotations.loc[q, f'annot_{d}'] == k) & (tmp_annotations.loc[q, f'id_{d}'] in u_include):
                    k_weight[k] +=1

        if not np.array_equal(np.zeros(car), k_weight):
            max_val = max(k_weight)
            max_indices = []
            for i, k in enumerate(k_weight):
                if k == max_val:
                    max_indices.append(i)
            tmp_annotations.loc[q, 'krip'] = max_indices[np.random.randint(max_indices.__len__())]
        else:
            tmp_annotations.loc[q, 'krip'] = np.nan


    # determine differences
    diff_k = tmp_annotations.loc[tmp_annotations['krip'].notnull(), 'GT'] - tmp_annotations.loc[tmp_annotations['krip'].notnull(), 'krip']

    # count differences
    diff_k_cnt = (diff_k != 0).sum()

    # proportion correct after pruning and pc corrected for missing items
    pc_aftr_prun =  (1 - (diff_k_cnt / sum(tmp_annotations['krip'].notnull())))
    pc_aftr_prun_total =  pc_aftr_prun*(np.count_nonzero(np.sum(users.notnull()))/tmp_annotations.__len__())
    return T_dist, (alpha_bfr_prun, n_annot_aftr_prun, alpha_aftr_prun, n_answ_aftr_prun, pc_aftr_prun, pc_aftr_prun_total)

def makeplaceholderframe(model, idx, datalen, cols):
    assert datalen % 1 == 0
    datalen = int(datalen)
    df = pandas.DataFrame(np.concatenate((np.full((1,datalen), idx),np.full((1,datalen), model))).T, columns = cols[:2])
    df[cols[2:]] = np.zeros((datalen, cols.__len__()-2))
    return df


def process_model(model, session_dir, sessions, session_len, data, cols, size, sweeps, sweeptype, car, dup, p_fo, kg_q, kg_u):
    with open(f'{session_dir}/{sessions[0]}/output/{model}_data.pickle', 'rb') as file:
        tmp_data = pickle.load(file)
    try:
        data.loc[(data['model'] == model) & (data['session'] == 'avg'),
        ['car', 'T_dist', 'sweeptype', 'dup', 'p_fo', 'kg_q','kg_u']] = np.array(tmp_data.loc[(tmp_data['size'] == size) & (tmp_data['car'] == car) & (tmp_data['kg_u'] == kg_u),
        ['car', 'T_dist', 'sweeptype', 'dup', 'p_fo','kg_q', 'kg_u']])
    except:
        print(f'process model failed on {[model, session_dir, sessions, data, cols, size, car, dup, p_fo, kg_q, kg_u]}')

    # fill frame with values
    for idx, session in enumerate(sessions):
        filepath = f'{session_dir}/{session}/output/{model}_data.pickle'
        with open(filepath, 'rb') as file:
            tmp_data = pickle.load(file)


        data.loc[(data['model'] == model) & (data['session'] == 'avg'),
            ['pc_m', 'pc_n', 'pc_n_KG', 'CertaintyQ', 'CertaintyA']] = data.loc[(data['model'] == model) & (data['session'] == 'avg'),
            ['pc_m', 'pc_n', 'pc_n_KG', 'CertaintyQ', 'CertaintyA']] + np.array(tmp_data.loc[:, ['pc_m', 'pc_n', 'pc_n_KG', 'CertaintyQ', 'CertaintyA']] / sessions.__len__())
        data = pandas.concat((data, makeplaceholderframe(model, idx, session_len, cols)), ignore_index=True)
        data.loc[(data['session'] == f'{idx}') & (data['model'] == f'{model}'),
        ['iterations', 'car', 'T_dist', 'dup', 'p_fo', 'kg_q', 'kg_u', 'OBJ', 'pc_m', 'pc_n', 'pc_n_KG']] = np.array(tmp_data.loc[:,
        ['iterations', 'car', 'T_dist', 'dup', 'p_fo', 'kg_q', 'kg_u', f'{model}', 'pc_m', 'pc_n', 'pc_n_KG']])


        # if model == 'mcmc': # only need to do krip calc once per dataset, no need to repeat same calc for em
        #     queue_alpha_uerror_metrics(model, session_dir, session, idx, data, sweeps[sweeptype], car, dup, p_fo, kg_q, kg_u)
    for T_dist in sweeps[sweeptype]:


        n_m_cert = 0
        n_q = 0

        # for q in tmp_data.loc[(tmp_data['T_dist'] == T_dist), model].item().model.questions.values():
        #     n_q += 1
        #     if np.exp(q.logProb()) > 0.9:
        #         n_m_cert += q.model == q.GT
        # pc_m_cert = n_m_cert / n_q
        pc_m_cert = 0 # not used
        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_m_cert'] = data.loc[(data['session'] == 'avg') &
                                                             (data['model'] == model) &
                                                             (data['car'] == car) &
                                                             (data['T_dist'] == T_dist) &
                                                             (data['dup'] == dup) &
                                                             (data['p_fo'] == p_fo) &
                                                             (data['kg_q'] == kg_q) &
                                                             (data['kg_u'] == kg_u), 'pc_m_cert'] + (pc_m_cert/sessions.__len__())

        # data.loc[(data['session'] == f'{idx}') & (data['model'] == f'{model}'), 'pc_m_cert'] = pc_m_cert

        # make a slice of all the sessions without the average
        dat = data.loc[(data['session'] != 'avg') &
                       (data['model'] == model) &
                       (data['car'] == car) &
                       (data['T_dist'] == T_dist) &
                       (data['dup'] == dup) &
                       (data['p_fo'] == p_fo) &
                       (data['kg_q'] == kg_q) &
                       (data['kg_u'] == kg_u)]
        T_diff = []
        if dat.__len__() != 10:
            print(f"unexpected number of sessions: {dat.__len__()}")
        n_annots = 0

        for k in dat['OBJ'].values: # for each session for this dist
            if model == 'em':
                for annot in list(k.annotators.values()):
                    prop_t = beta.mean(annot.a, annot.b)
                    T_diff.append(abs(annot.T - prop_t))
                n_annots += k.annotators.__len__()
            elif model == 'mcmc':
                for annot in list(k.model.annotators.values()):
                    prop_t = beta.mean(annot.posterior[0],annot.posterior[1])
                    T_diff.append(abs(annot.T - prop_t))
                n_annots += k.model.annotators.__len__()
            else:
                raise ValueError # always should be either mcmc or em





        # do ttest
        # ttest needs to be done only once per parameter configuration, so only do on mcmc round:
        if model == 'mcmc':
            pc_em = data.loc[(data['session'] != 'avg') &
                           (data['model'] == 'em') &
                           (data['car'] == car) &
                           (data['T_dist'] == T_dist) &
                           (data['dup'] == dup) &
                           (data['p_fo'] == p_fo) &
                           (data['kg_q'] == kg_q) &
                           (data['kg_u'] == kg_u), 'pc_m']

            pc_mcmc = data.loc[(data['session'] != 'avg') &
                           (data['model'] == 'mcmc') &
                           (data['car'] == car) &
                           (data['T_dist'] == T_dist) &
                           (data['dup'] == dup) &
                           (data['p_fo'] == p_fo) &
                           (data['kg_q'] == kg_q) &
                           (data['kg_u'] == kg_u), 'pc_m']
            res = ttest_ind(pc_mcmc, pc_em, alternative='greater')

            data.loc[(data['session'] == 'avg') &
                     (data['model'] == model) &
                     (data['car'] == car) &
                     (data['T_dist'] == T_dist) &
                     (data['dup'] == dup) &
                     (data['p_fo'] == p_fo) &
                     (data['kg_q'] == kg_q) &
                     (data['kg_u'] == kg_u), 'ttest_val'] = res.statistic
            data.loc[(data['session'] == 'avg') &
                     (data['model'] == model) &
                     (data['car'] == car) &
                     (data['T_dist'] == T_dist) &
                     (data['dup'] == dup) &
                     (data['p_fo'] == p_fo) &
                     (data['kg_q'] == kg_q) &
                     (data['kg_u'] == kg_u), 'ttest_p'] = res.pvalue


        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'T_diff'] = np.mean(T_diff)
        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'T_diff_SD'] = np.std(T_diff)

        # determine SD for maj. vote and model
        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_n_SD'] = np.std(dat['pc_n'])
        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_n_KG_SD'] = np.std(dat['pc_n_KG'])

        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_m_SD'] = np.std(dat['pc_m'])

        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_krip'] = np.mean(dat['pc_krip'])
        data.loc[(data['session'] == 'avg') &
                 (data['model'] == model) &
                 (data['car'] == car) &
                 (data['T_dist'] == T_dist) &
                 (data['dup'] == dup) &
                 (data['p_fo'] == p_fo) &
                 (data['kg_q'] == kg_q) &
                 (data['kg_u'] == kg_u), 'pc_krip_SD'] = np.std(dat['pc_krip'])
    return data

def find_params(session_dir):
    if platform.system() == 'Windows':
        properties = session_dir.split("\\")
    else:
        properties = session_dir.split("/")
    size = properties[1][properties[1].index("_")+1:]
    sweeptype = properties[2][properties[2].index("_")+1:]
    car = int(properties[3][properties[3].index("_")+1:])
    dup = int(properties[4][properties[4].index("_") + 1:])
    p_fo = float(properties[5][-3:])
    kg_q = int(properties[6][re.match('kg_q_', properties[6]).regs[0][1]:])
    kg_u = int(properties[7][re.match('kg_u_', properties[7]).regs[0][1]:])
    return size, sweeptype, car, dup, p_fo, kg_q, kg_u

def main(session_dir, step, sweeps):
    # latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'

    # if not os.path.exists(f'{session_dir}/stats.pickle'):
    if True:
        em_sessions = []
        mcmc_sessions = []
        session_len = 0
        print(session_dir)
        for dir in step[1]:
            types = next(os.walk(f'{session_dir}/{dir}/output'))[2]


            # try:
            type = types[0][:2]
            tpcnt = 0
            while type != 'mc' and type != 'em':
                tpcnt += 1
                type = types[tpcnt][:2]
            # except Exception as e:
            #     print(f'Incomplete session: {session_dir}/{dir}')
            #     print(e)
            #     continue
            if type == 'em':
                em_sessions.append(dir)
                session_len = int((next(os.walk(f'{session_dir}/{dir}/output'))[2].__len__() - 1) / 2)
            elif type == 'mc':
                mcmc_sessions.append(dir)
                session_len = int((next(os.walk(f'{session_dir}/{dir}/output'))[2].__len__()-1)/3)
            else:
                print(f"unexpected type: {type}")
                raise ValueError

            # initialize dataset

            # session_len = int((next(os.walk(f'{session_dir}/{dir}/output'))[2].__len__()-1)/2)
            if session_len != 11:
                print(f"WARNING number of entries in session was expected to be 11, but is {session_len} in folder {session_dir}/{dir}")
        assert session_len !=0
        datalen = 2*session_len # account for both EM and MCMC models

        size, sweeptype, car, dup, p_fo, kg_q, kg_u = find_params(session_dir)


        # session denotes the session number, all are needed in memory at once to calculate SD. Session 'avg' is the average over all sessions
        cols = ['session', 'model', 'car', 'sweeptype', 'T_dist', 'dup', 'p_fo', 'kg_q', 'kg_u', 'OBJ', 'pc_m', 'pc_m_cert', 'CertaintyQ',
                'CertaintyA', 'pc_m_SD', 'pc_n', 'pc_n_SD', 'pc_n_KG', 'pc_n_KG_SD', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun','n_answ_aftr_prun',
                'pc_aftr_prun', 'alpha_aftr_prun', 'pc_krip', 'pc_krip_SD', 'T_diff', 'T_diff_SD', 'ttest_val', 'ttest_p' ]
        data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
        data.loc[:datalen/2,'model'] = "em"
        data.loc[datalen / 2:, 'model'] = "mcmc"
        data.loc[:,'session'] = 'avg'

        if em_sessions.__len__()>0:
            data = process_model('em', session_dir, em_sessions, session_len, data, cols, size, sweeps, sweeptype, car, dup, p_fo, kg_q, kg_u)


        if mcmc_sessions.__len__()>0:
            data = process_model('mcmc', session_dir, mcmc_sessions, session_len, data, cols, size, sweeps, sweeptype, car, dup, p_fo, kg_q, kg_u)

        with open(f'{session_dir}/stats.pickle', 'wb') as file:
            pickle.dump(data, file)

