from collections import Counter
import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
from scipy.stats import beta
from datetime import datetime
import os
from create_simulation_data import createData

class mcmc():

    def __init__(self, K):
        self.M = np.arange(0,nQuestions)    # Questions
        self.L = np.arange(0,K)             # Given label per question
        self.K = K                          # Number of answer options
        self.cm = K-1                       # -1 because there's one good answer and the rest is wrong
        self.iter = 0

    def p_tn(self, user, annotations, i):
        """
        qs is a series of the questions answered by this annotator
        n_eq is the number of times l is equal to lhat
        returns expected value, alpha and beta
        """
        qs = user.loc[user['ID']==i, user.loc[user['ID']==i, :].notnull().squeeze()].squeeze()

        # define indices to take from qs
        startindex = 4
        endindex = qs.__len__()-2-self.iter

        n_eq = sum(np.equal(np.array(qs[startindex:endindex]),np.array(annotations.loc[[int(i[2:]) for i in qs.index[startindex:endindex]],'model'])))
        return n_eq/qs[startindex:endindex].__len__(), n_eq, qs[startindex:endindex].__len__()-n_eq

    def p_lhat_k(self, user, m, k):

        num = np.prod([
                            (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) #* beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                           for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                        ])
        denom = sum([
                    np.prod([
                            (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) # * beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                              for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ]
                    ) for l in self.L])

        y = num/denom
        return y

    def Gibbs_tn(self, user, annotations, priora, priorb, i):
        res = self.p_tn(user, annotations, i)
        user.loc[i,'a'] = float(res[1] + priora + np.spacing(0)) # np.spacing to prevent alpha == 0
        user.loc[i,'b'] = float(res[2] + priorb + np.spacing(0)) # np.spacing to prevent beta == 0
        return beta.rvs(user.loc[i,'a'],user.loc[i,'b'],size=1)[0]


    def Gibbs_lhat(self, user, annotations, i):
        if annotations.loc[i, 'KG'] == True:
            return annotations.loc[i, 'GT']
        else:
            p = [self.p_lhat_k(user, i, k) for k in self.L]
            return np.random.choice(self.K, p=p)

    def posterior(self, nQuestions, annotations):

        # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
        posteriorindices = (10 * [False])+[x % 3 == 0 for x in range(30)]

        # average modelled trustworthiness over selected samples
        for u in range(user.__len__()):
            user.loc[u, f'T_model'] = np.mean(user.loc[u, [f'T_model_{i}' for i, x in enumerate(posteriorindices) if x]])

        # count occurences in posterior to produce estimate
        for q in range(nQuestions):
            cnt = Counter(annotations.loc[q,[f'model_{i}' for i,x in enumerate(posteriorindices) if x]])
            mc = cnt.most_common(2)
            if mc.__len__() > 1:
                if mc[0][1] == mc[1][1]:
                    if mc.__len__() > 2:
                        if mc[1][1] == mc[2][1]:
                            print(f"values {mc[0][0]}, {mc[1][0]} and {mc[2][0]} are equally frequent")
                            estimate = np.random.randint(3)
                        else:
                            print(f"values {mc[0][0]} and {mc[1][0]} are equally frequent")
                            estimate = np.random.randint(2)
                    else:
                        print(f"values {mc[0][0]} and {mc[1][0]} are equally frequent")
                        estimate = np.random.randint(2)
                else:
                    estimate = 0
            else:
                estimate = 0
            annotations.loc[q, 'model'] = mc[estimate][0] # pick most common, which returns list of lists with value and occurences

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
        mcmc_data.loc[(mcmc_data['size'].values == size) &
                      (mcmc_data['iterations'].values == iterations) &
                      (mcmc_data['car'].values == car) &
                      (mcmc_data['mode'].values == mode) &
                      (mcmc_data['dup'].values == dup) &
                      (mcmc_data['p_fo'].values == p_fo) &
                      (mcmc_data['p_kg'].values == p_kg) &
                      (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
        mcmc_data.loc[(mcmc_data['size'].values == size) &
                      (mcmc_data['iterations'].values == iterations) &
                      (mcmc_data['car'].values == car) &
                      (mcmc_data['mode'].values == mode) &
                      (mcmc_data['dup'].values == dup) &
                      (mcmc_data['p_fo'].values == p_fo) &
                      (mcmc_data['p_kg'].values == p_kg) &
                      (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))


    def run(self, iterations, car, nQuestions, user, annotations):

        # run iterations
        while self.iter < iterations:
            if self.iter % 10 == 0:
                print("iteration: ", self.iter)

            # sample l_hat

            # first only the KG's, as that primes the lhats for the other samples with the right bias
            indices = annotations.loc[(annotations['KG']==True), 'ID']
            if indices.__len__()>0:
                with Pool(32) as p:
                    results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                          (mcmc_data['iterations'].values == iterations) &
                                                          (mcmc_data['car'].values == car) &
                                                          (mcmc_data['mode'].values == mode) &
                                                          (mcmc_data['dup'].values == dup) &
                                                          (mcmc_data['p_fo'].values == p_fo) &
                                                          (mcmc_data['p_kg'].values == p_kg) &
                                                          (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_lhat, user, annotations), indices)
                    annotations.loc[(annotations['KG']==True), 'model'] = results
                    annotations.loc[(annotations['KG']==True), f'model_{self.iter}'] = results

            # after the KG's, do the rest of the samples
            indices = annotations.loc[(annotations['KG'] == False), 'ID']
            with Pool(32) as p:
                results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                      (mcmc_data['iterations'].values == iterations) &
                                                      (mcmc_data['car'].values == car) &
                                                      (mcmc_data['mode'].values == mode) &
                                                      (mcmc_data['dup'].values == dup) &
                                                      (mcmc_data['p_fo'].values == p_fo) &
                                                      (mcmc_data['p_kg'].values == p_kg) &
                                                      (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_lhat, user, annotations), indices)
                annotations.loc[(annotations['KG']==False), 'model'] = results
                annotations.loc[(annotations['KG']==False), f'model_{self.iter}'] = results


            # sample tn



            # first do the KG users
            indices = user.loc[(user['type']=='KG'), 'ID']
            if indices.__len__()>0:
                with Pool(32) as p:
                    results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                          (mcmc_data['iterations'].values == iterations) &
                                                          (mcmc_data['car'].values == car) &
                                                          (mcmc_data['mode'].values == mode) &
                                                          (mcmc_data['dup'].values == dup) &
                                                          (mcmc_data['p_fo'].values == p_fo) &
                                                          (mcmc_data['p_kg'].values == p_kg) &
                                                          (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_tn, user, annotations, priora, priorb), indices)

                    user.loc[(user['type']=='KG'), 'T_model'] = results
                    user.loc[(user['type']=='KG'), f'T_model_{self.iter}'] = results

            # After KG users, do the rest
            indices = user.loc[(user['type'] != 'KG'), 'ID']
            with Pool(32) as p:
                results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                      (mcmc_data['iterations'].values == iterations) &
                                                      (mcmc_data['car'].values == car) &
                                                      (mcmc_data['mode'].values == mode) &
                                                      (mcmc_data['dup'].values == dup) &
                                                      (mcmc_data['p_fo'].values == p_fo) &
                                                      (mcmc_data['p_kg'].values == p_kg) &
                                                      (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_tn, user, annotations, priora, priorb), indices)

                user.loc[(user['type'] != 'KG'), 'T_model'] = results
                user.loc[(user['type'] != 'KG'), f'T_model_{self.iter}'] = results

            print_intermediate_results = False
            if print_intermediate_results:
                print(f'GT correct {sum(np.equal(np.array(annotations["model"]), np.array(annotations["GT"])))} out of {annotations.__len__()}')
                print(f"average Tn offset: {np.mean(np.abs(user['T_given']-user['T_model']))}")
                print(f"closeness: {sum(user['T_model'])/(sum(user['T_given'])+np.spacing(0))}")
            self.iter += 1

        self.posterior(nQuestions, annotations)

        summary = {"Mode": mode,
                   "Cardinality": car,
                   "Iterations": iterations,
                   "Duplication factor": dup,
                   "Proportion 'first only'": p_fo,
                   "Proportion 'known good'": p_kg,
                   "Percentage correct modelled": mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                                (mcmc_data['iterations'].values == iterations) &
                                                                (mcmc_data['car'].values == car) &
                                                                (mcmc_data['mode'].values == mode) &
                                                                (mcmc_data['dup'].values == dup) &
                                                                (mcmc_data['p_fo'].values == p_fo) &
                                                                (mcmc_data['p_kg'].values == p_kg) &
                                                                (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_m'],
                   "Percentage correct naive": mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                             (mcmc_data['iterations'].values == iterations) &
                                                             (mcmc_data['car'].values == car) &
                                                             (mcmc_data['mode'].values == mode) &
                                                             (mcmc_data['dup'].values == dup) &
                                                             (mcmc_data['p_fo'].values == p_fo) &
                                                             (mcmc_data['p_kg'].values == p_kg) &
                                                             (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_n']
                   }

        print_final_results = False
        if print_final_results:
            [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":



    iterations_list = [40]        # iterations of mcmc algorithm -- 10 warmup - keep 30 - sample 10 from these 30
    # car_list = list(range(2,8))     # cardinality of the questions
    # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    # dups = [3,5,7,9]                # duplication factor of the annotators
    # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    # p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]

    car_list = [9]
    modes = [f'single{round(flt,2)}' for flt in np.arange(0,1.1,0.1)]
    dups = [3]
    p_fos = [0.0, 0.1]
    p_kgs = [0.0, 0.1]
    p_kg_us = [0.0, 0.1]

    priora = 1
    priorb = 1
    session_dir = f'sessions/prior-{priora}_{priorb}-car{car_list[0]}/session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


    # os.makedirs(os.path.dirname(f'{os.getcwd()}/data/{session_folder}'), exist_ok=True)
    os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)
    createData(f'{session_dir}', car_list, modes, dups, p_fos, p_kg_us)

    # resume mode allows the loading of an mcmc_data dataframe to continue training after it has been stopped
    # if not resuming, makes new empty dataframe with correct columns
    resume_mode = False
    if resume_mode:
        with open(f'sessions/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
            mcmc_data = pickle.load(file)
    else:
        mcmc_data = pandas.DataFrame(
            columns=['size', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'mcmc', 'pc_m', 'pc_n'])

    for size in ['small','medium','large']:
        for iterations in iterations_list:
            for car in car_list:
                for mode in modes:
                    for dup in dups:
                        for p_fo in p_fos:
                            for p_kg in p_kgs:
                                for p_kg_u in p_kg_us:
                                    # open dataset for selected parameters
                                    with open(f'{session_dir}/simulation data/{mode}/pickle/{size}_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_user.pickle',
                                              'rb') as file:
                                        user = pickle.load(file)
                                    with open(
                                            f'{session_dir}/simulation data/{mode}/pickle/{size}_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_annotations_empty.pickle',
                                            'rb') as file:
                                        annotations = pickle.load(file)

                                    ## add parameters to dataframes
                                    # known goods
                                    annotations[f'KG'] = [np.random.choice([0,1], p=[1-p_kg,p_kg]) for _ in range(annotations.__len__())]
                                    # user[f'KG'] = [np.random.choice([0, 1], p=[1 - p_kg_u, p_kg_u]) for _ in
                                    #                       range(user.__len__())]

                                    # user parameters for beta sampling
                                    user['a'] = np.ones(user.__len__())
                                    user['b'] = np.ones(user.__len__())

                                    # trustworthiness over iterations
                                    ucols = []
                                    for i in range(iterations):
                                        ucols += [f't_weight_{i}']
                                    weights = pandas.DataFrame(np.ones((user.__len__(),iterations))*0.5, columns= ucols)
                                    pandas.concat((user, weights) )

                                    # global nQuestions
                                    nQuestions = annotations.__len__()

                                    # create mcmc_data dataframe
                                    mcmc_data.loc[mcmc_data.__len__(), :] = [size, iterations, car, mode, dup, p_fo, p_kg, p_kg_u, None, 0, 0]
                                    # init mcmc object
                                    mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                  (mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg) &
                                                  (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'] = mcmc(car)

                                    mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                  (mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg) &
                                                  (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].item().run(iterations, car, nQuestions, user, annotations)

                                    with open(f'{session_dir}/output/mcmc_annotations_data_size-{size}_mode-{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations}.pickle', 'wb') as file:
                                        pickle.dump(annotations, file)
                                    with open(f'{session_dir}/output/mcmc_user_data_size-{size}_mode-{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations}.pickle', 'wb') as file:
                                        pickle.dump(user, file)
                                    with open(f'{session_dir}/output/mcmc_data_size-{size}{"_".join(modes)}.pickle', 'wb') as file:
                                        pickle.dump(mcmc_data, file)