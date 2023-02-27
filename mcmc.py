import platform
from collections import Counter
import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
from scipy.stats import beta
from datetime import datetime
import os
from create_simulation_data import createData

from numpy.random import default_rng
rng = default_rng()

class mcmc():

    def __init__(self, K):
        self.M = np.arange(0,nQuestions)    # Questions
        self.L = np.arange(0,K)             # Given label per question
        self.K = K                          # Number of answer options
        self.cm = K-1                       # -1 because there's one good answer and the rest is wrong
        self.iter = 0

    def p_tn(self, user, annotations, i):
        """
        q_answered is a series of the questions answered by this annotator
        n_eq is the number of times l is equal to lhat
        returns expected value, alpha and beta
        """
        q_answered = user.loc[user['ID']==i, user.loc[user['ID']==i, :].notnull().squeeze()].squeeze()

        # define columns to take from q_answered
        startindex = 4
        endindex = np.where(q_answered.index=='a')[0][0]
        # endindex = q_answered.__len__()-2-self.iter

        n_eq = sum(
            np.equal(
                np.array(q_answered[startindex:endindex]),
                np.array(annotations.loc[[int(i[2:]) for i in q_answered.index[startindex:endindex]],'model'])
            )
        )
        expected_val =n_eq/q_answered[startindex:endindex].__len__()
        return expected_val, n_eq, q_answered[startindex:endindex].__len__()-n_eq

    def p_lhat_k(self, user, m, k):
        # T_model if answer for that user is k else 1-T_model/k-1, times 1/k
        num = np.prod([
                            (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) #* beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                           for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                        ])
        # 
        denom = sum([
                    np.prod([
                            (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) # * beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
                              for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ]
                    ) for l in self.L])

        y = num/denom
        return y

    def Gibbs_tn(self, user, annotations, priors, nSamples, i):
        if user.loc[i,'type'] == 'KG':
            return 1
        else:
            alpha, beta = priors['aAlpha'], priors['aBeta']
            q_answered = user.loc[user['ID'] == i, user.loc[user['ID'] == i, :].notnull().squeeze()].squeeze() # this can be defined nicer using the annotations dataframe

            # define columns to take from q_answered
            startindex = 4
            endindex = np.where(q_answered.index == 'a')[0][0]

            for val, idx in zip(q_answered[startindex: endindex], q_answered.index[startindex:endindex]):
                val = int(val)
                idx = int(idx[2:])
                for _ in range(nSamples):
                    qAlpha = annotations.loc[idx, 'alpha']
                    v = rng.dirichlet(qAlpha)
                    t = rng.beta(user.loc[i, 'a'], user.loc[i, 'b'])
                    pos = t*v[val]  # * np.prod(1.-np.concatenate((v[:a.value],v[(a.value+1):])))
                    neg = (1.-t) / annotations.loc[idx, 'car']  # * (1.-a.question.cardinality)**(a.question.cardinality-1)
                    #                debug(v,v[a.value],pos,neg)
                    post = pos / (pos + neg)
                    alpha += post/nSamples
                    beta += (1. - post)/nSamples
            return alpha, beta

            # res = self.p_tn(user, annotations, i)
            # user.loc[i,'a'] = float(res[1] + priors['aAlpha'])
            # user.loc[i,'b'] = float(res[2] + priors['aBeta'])
            # return rng.beta(user.loc[i,'a'],user.loc[i,'b'],1)[0]
            # return beta.rvs(user.loc[i,'a'],user.loc[i,'b'],size=1)[0]

    def Gibbs_lhat(self, user, annotations, priors, nSamples, i):
        if annotations.loc[i, 'KG'] == True:
            # if we know this annotation to be correct, we don't need to sample
            return annotations.loc[i, 'GT']
        elif "KG" in user.loc[~np.isnan(user.loc[:,f"q_{i}"]), 'type']:
            # is we know at least one of the annotators is good, we don't need to sample
            # this should really be: if we have a KG, take the answer from the person, but this is the same as taking GT
            return annotations.loc[i, 'GT']
        else:
            alpha = tuple((priors['aAlpha'] for _ in range(self.K))) # implement some l.c cardinaty per annotation
            for n in user.loc[~np.isnan(user.loc[:, f"q_{i}"])].iterrows():
                for _ in range(nSamples):
                    t = rng.beta(n[1]['a'], n[1]['b'])
                    alpha += (t * np.eye(self.K)[int(n[1][f"q_{i}"])])/nSamples
            return alpha

            # self.posterior = alpha
            # p = [self.p_lhat_k(user, i, k) for k in self.L]
            # return np.random.choice(self.K, p=p)

    def process_pc_posterior(self, nQuestions, annotations):

        # average modelled trustworthiness over selected samples
        for u in range(user.__len__()):
            user.loc[u, f'T_model'] = np.mean(user.loc[u, [f'T_model_{i}' for i in range(keep_n_samples)]])

        for q in range(nQuestions):
            alphas = annotations.loc[q, [f'alpha_{i}' for i in range(keep_n_samples)]]
            p = rng.dirichlet(np.mean(alphas.T))
            annotations.loc[q, 'model'] = np.where(rng.multinomial(1, p) ==1)[0][0]

        # count occurences in posterior to produce estimate
        # for q in range(nQuestions):
        #     cnt = Counter(annotations.loc[q,[f'alpha_{i}' for i in range(keep_n_samples)]])
        #     mc = cnt.most_common()
        #
        #     # pick a random option when the number of drawn samples is equal an option k
        #     max_entries = []
        #     for k, n in mc:
        #         if n == mc[0][1]:
        #             max_entries.append(k)
        #     annotations.loc[q, 'model']  = max_entries[np.random.randint(max_entries.__len__())]

        # do naive estimation for baseline
        # insert new empty column
        annotations.insert(annotations.columns.get_loc("model") + 1, "naive", np.zeros(nQuestions))
        for q in range(nQuestions):
            # weights for all k options list
            k_w = []
            for k in range(car):
                # counter for number of people who chose option k
                d_w = 0
                for d in range(dup):
                    if annotations.loc[q, f'annot_{d}'] == k:
                        d_w += 1
                k_w.append(d_w)
            max_val = max(k_w)
            max_indices = []
            for i, k in enumerate(k_w):
                if k == max_val:
                    max_indices.append(i)
            annotations.loc[q, 'naive'] = max_indices[np.random.randint(max_indices.__len__())]

        # determine differences
        diff_m = annotations.loc[:, 'GT'] - annotations.loc[:, 'model']
        diff_n = annotations.loc[:, 'GT'] - annotations.loc[:, 'naive']

        # count differences
        diff_m_cnt = (diff_m != 0).sum()
        diff_n_cnt = (diff_n != 0).sum()

        # insert into mcmc_data dataframe
        mcmc_data.loc[(mcmc_data['size'].values == size) &
                      (mcmc_data['iterations'].values == keep_n_samples) &
                      (mcmc_data['car'].values == car) &
                      (mcmc_data['mode'].values == T_dist) &
                      (mcmc_data['dup'].values == dup) &
                      (mcmc_data['p_fo'].values == p_fo) &
                      (mcmc_data['p_kg'].values == p_kg) &
                      (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
        mcmc_data.loc[(mcmc_data['size'].values == size) &
                      (mcmc_data['iterations'].values == keep_n_samples) &
                      (mcmc_data['car'].values == car) &
                      (mcmc_data['mode'].values == T_dist) &
                      (mcmc_data['dup'].values == dup) &
                      (mcmc_data['p_fo'].values == p_fo) &
                      (mcmc_data['p_kg'].values == p_kg) &
                      (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))


    def run(self, iterations, car, nQuestions, user, annotations, priors):

        # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
        posteriorindices = (warmup * [False])+[x % sample_interval == 0 for x in range(iterations*sample_interval)]

        # counter to keep track of how many samples are taken
        sample_cnt = 0

        # run iterations
        while self.iter < posteriorindices.__len__():
            # if self.iter % 10 == 0:
            print("iteration: ", self.iter)

            ## sample l_hat
            # first only the KG's, as that primes the lhats for the other samples with the right bias
            KG_ids = annotations.loc[(annotations['KG']==True), 'ID']
            if KG_ids.__len__()>0:
                with Pool(16) as p:
                    results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                          (mcmc_data['iterations'].values == iterations) &
                                                          (mcmc_data['car'].values == car) &
                                                          (mcmc_data['mode'].values == T_dist) &
                                                          (mcmc_data['dup'].values == dup) &
                                                          (mcmc_data['p_fo'].values == p_fo) &
                                                          (mcmc_data['p_kg'].values == p_kg) &
                                                          (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_lhat, user, annotations, priors, nSamples), KG_ids)
                    annotations.loc[(annotations['KG']==True), 'alpha'] = pandas.Series(results)
                    if posteriorindices[self.iter]:
                        annotations.loc[(annotations['KG']==True), f'alpha_{sample_cnt}'] = pandas.Series(results)

            # after the KG's, do the rest of the samples
            indices = annotations.loc[(annotations['KG'] == False), 'ID']
            with Pool(16) as p:
                results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                      (mcmc_data['iterations'].values == iterations) &
                                                      (mcmc_data['car'].values == car) &
                                                      (mcmc_data['mode'].values == T_dist) &
                                                      (mcmc_data['dup'].values == dup) &
                                                      (mcmc_data['p_fo'].values == p_fo) &
                                                      (mcmc_data['p_kg'].values == p_kg) &
                                                      (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_lhat, user, annotations, priors, nSamples), indices)
                annotations.loc[(annotations['KG']==False), 'alpha'] = pandas.Series(results)
                if posteriorindices[self.iter]:
                    annotations.loc[(annotations['KG']==False), f'alpha_{sample_cnt}'] = pandas.Series(results)

            ## sample tn
            # first do the KG users
            indices = user.loc[(user['type']=='KG'), 'ID']
            if indices.__len__()>0:

                # no need to sample known good users: they are known good and therefore T = 1
                user.loc[(user['type']=='KG'), ['a','b']] = [1, np.spacing(1)]
                if posteriorindices[self.iter]:
                    user.loc[(user['type']=='KG'), f'T_model_{sample_cnt}'] = np.ones(indices.__len__())

            # After KG users, do the rest
            indices = user.loc[(user['type'] != 'KG'), 'ID']
            with Pool(16) as p:
                results = p.map(partial(mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                      (mcmc_data['iterations'].values == iterations) &
                                                      (mcmc_data['car'].values == car) &
                                                      (mcmc_data['mode'].values == T_dist) &
                                                      (mcmc_data['dup'].values == dup) &
                                                      (mcmc_data['p_fo'].values == p_fo) &
                                                      (mcmc_data['p_kg'].values == p_kg) &
                                                      (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].values[0].Gibbs_tn, user, annotations, priors, nSamples), indices)

                user.loc[(user['type'] != 'KG'), ['a','b']] = results
                if posteriorindices[self.iter]:
                    user.loc[(user['type'] != 'KG'), f'T_model_{sample_cnt}'] = [res[0]/(res[0]+res[1]) for res in results]

            print_intermediate_results = False
            if print_intermediate_results:
                print(f'GT correct {sum(np.equal(np.array(annotations["alpha"]), np.array(annotations["GT"])))} out of {annotations.__len__()}')
                print(f"average Tn offset: {np.mean(np.abs(user['T_given']-user['T_model']))}")
                print(f"closeness: {sum(user['T_model'])/(sum(user['T_given'])+np.spacing(0))}")
            if posteriorindices[self.iter]:
                sample_cnt += 1
            self.iter += 1

        # print(sample_cnt)
        assert(sample_cnt == iterations)

        self.process_pc_posterior(nQuestions, annotations)

        summary = {"Mode": T_dist,
                   "Cardinality": car,
                   "Iterations": iterations,
                   "Duplication factor": dup,
                   "Proportion 'first only'": p_fo,
                   "Proportion 'known good'": p_kg,
                   "Percentage correct modelled": mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                                (mcmc_data['iterations'].values == iterations) &
                                                                (mcmc_data['car'].values == car) &
                                                                (mcmc_data['mode'].values == T_dist) &
                                                                (mcmc_data['dup'].values == dup) &
                                                                (mcmc_data['p_fo'].values == p_fo) &
                                                                (mcmc_data['p_kg'].values == p_kg) &
                                                                (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_m'],
                   "Percentage correct naive": mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                             (mcmc_data['iterations'].values == iterations) &
                                                             (mcmc_data['car'].values == car) &
                                                             (mcmc_data['mode'].values == T_dist) &
                                                             (mcmc_data['dup'].values == dup) &
                                                             (mcmc_data['p_fo'].values == p_fo) &
                                                             (mcmc_data['p_kg'].values == p_kg) &
                                                             (mcmc_data['p_kg_u'].values == p_kg_u), 'pc_n']
                   }

        print_final_results = False
        if print_final_results:
            [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":

    ## settings

    # n samples to keep
    keep_samples_list = [100]

    # keep a sample every sample_interval iterations
    sample_interval = 20

    # warmup
    warmup = 50
    nSamples = 10

    # car_list = list(range(2,8))     # cardinality of the questions
    # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    # dups = [3,5,7,9]                # duplication factor of the annotators
    # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    # p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]

    car_list = [3]
    T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
    dup_list = [3]
    p_fo_list = [0.0]
    p_kg_list = [0.0]
    p_kg_u_list = [0.0]

    priors = {'qAlpha':1e-5,
              'aAlpha':1e-1,
              'aBeta':1e-5}

    session_dir = f'sessions/prior-{priors["aAlpha"]}_{priors["aBeta"]}-car{car_list[0]}/session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)
    if not platform.system() == 'Windows':
        createData(f'{session_dir}', car_list, T_dist_list, dup_list, p_fo_list, p_kg_u_list)


    # resume mode allows the loading of an mcmc_data dataframe to continue training after it has been stopped
    # if not resuming, makes new empty dataframe with correct columns
    resume_mode = False
    if resume_mode:
        with open(f'sessions/mcmc_data_{"_".join(T_dist_list)}.pickle', 'rb') as file:
            mcmc_data = pickle.load(file)
    else:
        mcmc_data = pandas.DataFrame(
            columns=['size', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'mcmc', 'pc_m', 'pc_n'])

    for size in ['small']: # multiple sizes are available: ['small','medium','large']
        for keep_n_samples in keep_samples_list:
            for car in car_list:
                for T_dist in T_dist_list:
                    for dup in dup_list:
                        for p_fo in p_fo_list:
                            for p_kg in p_kg_list:
                                for p_kg_u in p_kg_u_list:
                                    # open dataset for selected parameters
                                    with open(f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_user.pickle',
                                              'rb') as file:
                                        user = pickle.load(file)
                                    with open(
                                            f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_annotations_empty.pickle',
                                            'rb') as file:
                                        annotations = pickle.load(file)

                                    ## add parameters to dataframes
                                    # known goods
                                    annotations[f'KG'] = [np.random.choice([0,1], p=[1-p_kg,p_kg]) for _ in range(annotations.__len__())]

                                    # user parameters for beta sampling
                                    user['a'] = np.ones(user.__len__())
                                    user['b'] = np.ones(user.__len__())

                                    # trustworthiness over iterations
                                    ucols = []
                                    for i in range(keep_n_samples):
                                        ucols += [f't_weight_{i}']
                                    weights = pandas.DataFrame(np.ones((user.__len__(), keep_n_samples)) * 0.5, columns= ucols)
                                    pandas.concat((user, weights) )

                                    # global nQuestions
                                    nQuestions = annotations.__len__()

                                    # create mcmc_data dataframe
                                    mcmc_data.loc[mcmc_data.__len__(), :] = [size, keep_n_samples, car, T_dist, dup, p_fo, p_kg, p_kg_u, None, 0, 0]
                                    # init mcmc object
                                    mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                  (mcmc_data['iterations'].values == keep_n_samples) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == T_dist) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg) &
                                                  (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'] = mcmc(car)

                                    # run mcmc sampling
                                    mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                  (mcmc_data['iterations'].values == keep_n_samples) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == T_dist) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg) &
                                                  (mcmc_data['p_kg_u'].values == p_kg_u), 'mcmc'].item().run(keep_n_samples, car, nQuestions, user, annotations, priors)

                                    # save data
                                    with open(f'{session_dir}/output/mcmc_annotations_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(annotations, file)
                                    with open(f'{session_dir}/output/mcmc_user_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(user, file)
                                    with open(f'{session_dir}/output/mcmc_data_size-{size}{"_".join(T_dist_list)}.pickle', 'wb') as file:
                                        pickle.dump(mcmc_data, file)