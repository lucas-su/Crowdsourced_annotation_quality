import sys
from collections import Counter

import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
from scipy.stats import beta
from datetime import datetime
import os

class mcmc():

    def __init__(self, K):
        self.M = np.arange(0,nQuestions)    # Questions
        self.L = np.arange(0,K)             # Given label per question
        self.K = K                          # Number of answer options
        self.cm = K-1                       # -1 because there's one good answer and the rest is wrong

    def p_tn(self, user, annotations, i):
        """
        qs is a series of the questions answered by this annotator
        n_eq is the number of times l is equal to lhat
        returns expected value, alpha and beta
        """
        qs = user.loc[user['ID']==i, user.loc[user['ID']==i, :].notnull().squeeze()].squeeze()
        endindex = -(qs.__len__()-4-self.M.__len__())
        n_eq = sum(np.equal(np.array(qs[4:endindex]),np.array(annotations.loc[[int(i[2:]) for i in qs.index[4:endindex]],'model'])))
        return n_eq/qs[4:endindex].__len__(), n_eq, qs[4:endindex].__len__()-n_eq

    def p_lhat_k(self, user, m, k):

        num = np.prod([
                            (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                           for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                        ])
        denom = sum([
                    np.prod([
                            (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K)  * beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                              for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ]
                    ) for l in self.L])

        y = num/denom
        return y

    def Gibbs_tn(self, user, annotations, i):
        res = self.p_tn(user, annotations, i)
        user.loc[i,'a'] = float(res[1]+ np.spacing(0)) # prevent alpha == 0
        user.loc[i,'b'] = float(res[2]+ np.spacing(0)) # prevent beta == 0
        return beta.rvs(user.loc[i,'a'],user.loc[i,'b'],size=1)[0]


    def Gibbs_lhat(self, user, annotations, i):
        if annotations.loc[i, 'KG'] == True:
            return annotations.loc[i, 'GT']
        else:
            p = [self.p_lhat_k(user, i, k) for k in self.L]
            return np.random.choice(self.K, p=p)


def run_mcmc(iterations, car, nQuestions, user, annotations):

    # init mcmc object
    mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                  (mcmc_data['car'].values == car) &
                  (mcmc_data['mode'].values == mode) &
                  (mcmc_data['dup'].values == dup) &
                  (mcmc_data['p_fo'].values == p_fo) &
                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'] = mcmc(car)

    # run iterations
    i = 0
    while i < iterations:
        if i%10==0:
            print("iteration: ", i)

        with Pool(32) as p:
            results = p.map(partial(mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'].values[0].Gibbs_lhat, user, annotations), range(annotations.__len__()))


            annotations.loc[:, 'model'] = results
            annotations.loc[:, f'model_{i}'] = results

        with Pool(32) as p:
            results = p.map(partial(mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'].values[0].Gibbs_tn, user, annotations), user['ID'])

            user.loc[:, 'T_model'] = results
            user.loc[:, f'T_model_{i}'] = results

        print_intermediate_results = False
        if print_intermediate_results:
            print(f'GT correct {sum(np.equal(np.array(annotations["model"]), np.array(annotations["GT"])))} out of {annotations.__len__()}')
            print(f"average Tn offset: {np.mean(np.abs(user['T_given']-user['T_model']))}")
            print(f"closeness: {sum(user['T_model'])/(sum(user['T_given'])+np.spacing(0))}")
        i += 1

    # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
    posteriorindices = (10*[False])+[x%3==0 for x in range(30)]

    # count occurences in posterior to produce estimate
    for q in range(nQuestions):
        cnt = Counter(annotations.loc[q,[f'model_{i}' for i,x in enumerate(posteriorindices) if x]])
        mc = cnt.most_common(2)
        if mc.__len__() > 1:
            if mc[0][1] == mc[1][1]:
                print(f"values {mc[0][0]} and {mc[1][0]} are equally frequent")
        annotations.loc[q, 'model'] = mc[0][0] # pick most common, which returns list of lists with value and occurences

    # do naive estimation for baseline
    annotations.insert(annotations.columns.get_loc("model") + 1, "naive", np.zeros(nQuestions))
    for q in range(nQuestions):
        k_w = []
        for k in range(car):
            d_w = 0
            for d in range(dup):
                if annotations.loc[q, f'annot_{d}'] == k:
                    d_w += 1
            k_w.append(d_w)
        annotations.loc[q, 'naive'] = k_w.index(max(k_w))

    # determine differences
    diff_m = annotations.loc[:, 'GT'] - annotations.loc[:, 'model']
    diff_n = annotations.loc[:, 'GT'] - annotations.loc[:, 'naive']

    # count differences
    diff_m_cnt = (diff_m != 0).sum()
    diff_n_cnt = (diff_n != 0).sum()

    # insert into mcmc_data dataframe
    mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                  (mcmc_data['car'].values == car) &
                  (mcmc_data['mode'].values == mode) &
                  (mcmc_data['dup'].values == dup) &
                  (mcmc_data['p_fo'].values == p_fo) &
                  (mcmc_data['p_kg'].values == p_kg), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
    mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                  (mcmc_data['car'].values == car) &
                  (mcmc_data['mode'].values == mode) &
                  (mcmc_data['dup'].values == dup) &
                  (mcmc_data['p_fo'].values == p_fo) &
                  (mcmc_data['p_kg'].values == p_kg), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))

    summary = {"Mode": mode,
               "Cardinality": car,
               "Iterations": iterations,
               "Duplication factor": dup,
               "Proportion 'first only'": p_fo,
               "Proportion 'known good'": p_kg,
               "Percentage correct modelled": 100 * (1 - (diff_m_cnt / nQuestions)),
               "Percentage correct naive": 100 * (1 - (diff_n_cnt / nQuestions))}

    print_final_results = False
    if print_final_results:
        [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":
    session_folder = f'session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}\\'
    os.makedirs(os.path.dirname(f'{os.getcwd()}/data/{session_folder}'), exist_ok=True)

    iterations_list = [40]        # iterations of mcmc algorithm -- 10 warmup - keep 30 - sample 10 from these 30
    car_list = list(range(2,8))     # cardinality of the questions
    modes = ['uniform', 'gaussian', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # resume mode allows the loading of an mcmc_data dataframe to continue training after it has been stopped
    # if not resuming, makes new empty dataframe with correct columns
    resume_mode = False
    if resume_mode:
        with open(f'data/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
            mcmc_data = pickle.load(file)
    else:
        mcmc_data = pandas.DataFrame(
            columns=['iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'mcmc', 'pc_m', 'pc_n'])

    # small dataset for quick debugging
    if sys.argv[-1] == 'small_dataset':
        mode = 'gaussian'
        dup = 5
        p_fo = 0.2
        p_kg = 0.2
        car = 3
        iterations = 40
        with open(f'simulation data/small_test_user.pickle',
                  'rb') as file:
            user = pickle.load(file)
        with open(
                f'simulation data/small_test_annotations_empty.pickle',
                'rb') as file:
            annotations = pickle.load(file)
        annotations[f'KG'] = [np.random.choice([0, 1], p=[1 - p_kg, p_kg]) for _ in range(annotations.__len__())]
        user[f'T_prop'] = np.ones(user.__len__())
        user['a'] = np.ones(user.__len__())
        user['b'] = np.ones(user.__len__())
        ucols = []
        for i in range(iterations):
            ucols += [f't_weight_{i}']
        weights = pandas.DataFrame(np.ones((user.__len__(), iterations)) * 0.5, columns=ucols)
        pandas.concat((user, weights))
        user['included'] = np.ones(user.__len__())
        nQuestions = annotations.__len__()
        mcmc_data.loc[mcmc_data.__len__(), :] = [iterations, car, mode, dup, p_fo, p_kg, None, 0, 0]

        run_mcmc(iterations, car, nQuestions, user, annotations)
        with open(f'data/{session_folder}/mcmc_annotations_small_test.pickle', 'wb') as file:
            pickle.dump(annotations, file)
    else:
        for iterations in iterations_list:
            for car in car_list:
                for mode in modes:
                    for dup in dups:
                        for p_fo in p_fos:
                            for p_kg in p_kgs:
                                # open dataset for selected parameters
                                with open(f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_user.pickle',
                                          'rb') as file:
                                    user = pickle.load(file)
                                with open(
                                        f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_annotations_empty.pickle',
                                        'rb') as file:
                                    annotations = pickle.load(file)
                                annotations[f'KG'] = [np.random.choice([0,1], p=[1-p_kg,p_kg]) for _ in range(annotations.__len__())]
                                user[f'T_prop'] = np.ones(user.__len__())
                                user['a'] = np.ones(user.__len__())
                                user['b'] = np.ones(user.__len__())
                                ucols = []
                                for i in range(iterations):
                                    ucols += [f't_weight_{i}']
                                weights = pandas.DataFrame(np.ones((user.__len__(),iterations))*0.5, columns= ucols)
                                pandas.concat((user, weights) )
                                user['included'] = np.ones(user.__len__())
                                nQuestions = annotations.__len__()
                                mcmc_data.loc[mcmc_data.__len__(), :] = [iterations, car, mode, dup, p_fo, p_kg, None, 0, 0]

                                run_mcmc(iterations, car, nQuestions, user, annotations)
                                with open(f'data/{session_folder}/mcmc_annotations_p_kg-{p_kg}_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_iters-{iterations}.pickle', 'wb') as file:
                                    pickle.dump(annotations, file)
                                with open(f'data/{session_folder}/mcmc_user_p_kg-{p_kg}_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_iters-{iterations}.pickle', 'wb') as file:
                                    pickle.dump(user, file)
                                with open(f'data/{session_folder}/mcmc_data_{"_".join(modes)}.pickle', 'wb') as file:
                                    pickle.dump(mcmc_data, file)


