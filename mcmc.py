"""
some references
em:
https://github.com/RafaeNoor/Expectation-Maximization/blob/master/EM_Clustering.ipynb
https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
https://www.jstor.org/stable/pdf/2346806.pdf?refreqid=excelsior%3A02dbe84713a99816418f5ddd3b41a93c&ab_segments=&origin=&acceptTC=1

annotator reliability:
https://dl.acm.org/doi/pdf/10.1145/1743384.1743478
https://aclanthology.org/D08-1027.pdf
https://aclanthology.org/P99-1032.pdf
https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.0006-341X.2004.00187.x

https://www.aaai.org/ocs/index.php/WS/AAAIW12/paper/view/5350/5599

OG file:
https://colab.research.google.com/drive/186F0yeqV_5OpIC4FkjHysEJ0tAiEW6Y-?authuser=1#scrollTo=XTLv7eS2NORn
"""
import math
import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
from scipy.stats import beta
import time

# def timeit(func):
#     """
#     Decorator for measuring function's running time.
#     """
#     def measure_time(*args, **kw):
#         start_time = time.time()
#         result = func(*args, **kw)
#         print("Processing time of %s(): %.2f seconds."
#               % (func.__qualname__, time.time() - start_time))
#         return result
#
#     return measure_time


class mcmc():

    def __init__(self, K, iters):
        self.N = user['ID'] # annotators
        self.M = np.arange(0,nQuestions) # questions
        self.L = np.arange(0,K) # given label per question
        self.K = K
        self.cm = K-1 # -1 because there's one good answer and the rest is wrong
        # self.gamma_ = pandas.DataFrame(columns=[i for i in range(K)])
        self.iterations= iters

    # def transition_model(self,x):
    #     return np.random.normal(x, 0.3, (1,))[0]


    def prior(self,tn):
        # returns 1 for all valid values of tn. Log(1) =0, so it does not affect the summation.
        # returns 0 for all invalid values of tn (<=0 or >=1). Log(0)=-infinity, and Log(negative number) is undefined.
        # It makes the new tn infinitely unlikely.
        if (0<tn<1):
            return 1
        return np.spacing(0) # use spacing instead of 0 to prevent divide by zero

    # def acceptance(self, x, x_new):
    #     if x_new > x:
    #         return True
    #     else:
    #         # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
    #         # less likely x_new are less likely to be accepted
    #         return (np.random.uniform(0, 1) < (np.exp(x_new - x)))


    # def gamma(self, x, a, b):
    #     return (math.gamma(a + b)*x**(a-1)*(1-x)**(b-1)) / (math.gamma(a) * math.gamma(b))

    def p_tn(self, user, annotations, i):


        # qs is a series of the questions answered by this annotator
        # n_eq is the number of times l is equal to lhat
        # returns expected value, alpha and beta
        qs = user.loc[user['ID']==i, user.loc[user['ID']==i, :].notnull().squeeze()].squeeze()
        n_eq = sum(np.equal(np.array(qs[4:-4]),np.array(annotations.loc[[int(i[2:]) for i in qs.index[4:-4]],'model'])))
        return n_eq/qs[4:-4].__len__(), n_eq, qs[4:-4].__len__()-n_eq

    def p_lhat_k(self, user, annotation, m, k):
        # y = np.prod([
        #     n['T_model']/self.K for n in user.loc[~np.isnan(user.loc[:, f"q_{i}"])].iterrows()
        # ])/np.prod([
        #     (n['T_model'] / self.K)+((1-n['T_model'])/(self.K - 1)) for n in user.loc[~np.isnan(user.loc[:, f"q_{i}"])].iterrows()
        # ])

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
        # user.loc[i,'a'] = user.loc[i,'a'] + res[1]
        # user.loc[i,'b'] = user.loc[i,'b'] + res[2]
        user.loc[i,'a'] = float(res[1]+ np.spacing(0)) # prevent alpha == 0
        user.loc[i,'b'] = float(res[2]+ np.spacing(0)) # prevent beta == 0
        return beta.rvs(user.loc[i,'a'],user.loc[i,'b'],size=1)[0]


    def Gibbs_lhat(self, user, annotations, i):
        if annotations.loc[i, 'KG'] == True:
            return annotations.loc[i, 'GT']
        else:
            p = [self.p_lhat_k(user, annotations, i, k) for k in self.L]
            return np.random.choice(self.K, p=p)


    # def MH(self, user, annotations, n):
    #     # n is a single row of user df
    #
    #     x_new = self.transition_model(n[1]['T_model'])
    #     x_lik = self.p_D_theta(user, annotations, n, n[1]['T_model'])
    #     x_new_lik = self.p_D_theta(user, annotations, n, x_new)
    #     if self.acceptance(x_lik + np.log(self.prior(n[1]['T_model'])), x_new_lik + np.log(self.prior(x_new))):
    #         return 1, x_new, n[1]['T_model']
    #     else:
    #         return 0, x_new, n[1]['T_model']

def run_mcmc(iterations, car, nQuestions, user, annotations):
    mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                  (mcmc_data['car'].values == car) &
                  (mcmc_data['mode'].values == mode) &
                  (mcmc_data['dup'].values == dup) &
                  (mcmc_data['p_fo'].values == p_fo) &
                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'] = mcmc(car, iterations)
    i = 0

    # accepted = []
    # rejected = []
    # acCols = []
    # for n in range(user.__len__()):
    #     acCols += [f'n_{n}A', f'n_{n}R']
    # acceptionDF = pandas.DataFrame(columns=acCols)

    while i < iterations:
        print("iteration: ", i)

        # g = mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
        #                   (mcmc_data['car'].values == car) &
        #                   (mcmc_data['mode'].values == mode) &
        #                   (mcmc_data['dup'].values == dup) &
        #                   (mcmc_data['p_fo'].values == p_fo), 'mcmc'].values[0].e_step()
        # a,b= [2,2]
        with Pool(16) as p:
            results = p.map(partial(mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'].values[0].Gibbs_lhat, user, annotations), range(annotations.__len__()))


            annotations.loc[:, 'model'] = results

        with Pool(16) as p:
            results = p.map(partial(mcmc_data.loc[(mcmc_data['iterations'].values == iterations) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == mode) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg), 'mcmc'].values[0].Gibbs_tn, user, annotations), user['ID'])

            # T_beta = beta.pdf(results, a,b)*100/sum(beta.pdf(results, a,b))
            # user.loc[:,'T_model']= (T_beta+np.spacing(3))/(1+np.spacing(3))
            # user.loc[:, 'T_model'] = np.array(results)[:,0]
            # a = sum(np.array(results)[:,1])
            # b = sum(np.array(results)[:,2])
            user.loc[:, 'T_model'] = results

            # a = prop correct, b = total annotations


        # accepted = [result[1] if result[0]==1 else np.nan for result in results ]
        # rejected = [result[1] if result[0]==0 else np.nan for result in results ]
        # acceptionDF.loc[acceptionDF.__len__(), :] = np.array(list(zip(accepted,rejected))).flatten()
        # user.loc[:, "T_model"] = results
        # user.loc[:, f"t_weight_{i}"] = results
        print(f'GT correct {sum(np.equal(np.array(annotations["model"]), np.array(annotations["GT"])))} out of {annotations.__len__()}')
        # print(sum(user['T_model']))
        print(f"average Tn offset: {np.mean(np.abs(user['T_given']-user['T_model']))}")
        print(f"closeness: {sum(user['T_model'])/(sum(user['T_given'])+np.spacing(0))}")
        i += 1
    for q in range(nQuestions):
        k_w = np.zeros(car)
        for k in range(car):

            for d in range(dup):
                if annotations.loc[q, f'annot_{d}'] == k:
                    k_w = [k_w[i] + ((1 - user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                           k_w[i] + user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
                    # k_w[k] += user.loc[annotations.loc[q, f'id_{d}'], 'T_model']
                else:
                    # k_w = [k_w[i]+((1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'])/(car-1)) if i!= k else k_w[i] for i in range(car)]
                    k_w = [
                        k_w[i] + ((user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                        k_w[i] + 1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
        annotations.loc[q, 'model'] = k_w.index(max(k_w))
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

    diff_m = annotations.loc[:, 'GT'] - annotations.loc[:, 'model']
    diff_n = annotations.loc[:, 'GT'] - annotations.loc[:, 'naive']
    diff_m_cnt = (diff_m != 0).sum()
    diff_n_cnt = (diff_n != 0).sum()
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

    [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":

    iterations_list = [5,20]        # iterations of mcmc algorithm
    car_list = list(range(2,8))     # cardinality of the questions
    modes = ['uniform', 'gaussian', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    resume_mode = False
    if resume_mode:
        with open(f'data/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
            mcmc_data = pickle.load(file)
    else:
        mcmc_data = pandas.DataFrame(
            columns=['iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'mcmc', 'pc_m', 'pc_n'])
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
                            with open(f'data/mcmc_user_p_kg-{p_kg}_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_iters-{iterations}.pickle', 'wb') as file:
                                pickle.dump(user, file)
                            with open(f'data/mcmc_data_{"_".join(modes)}.pickle', 'wb') as file:
                                pickle.dump(mcmc_data, file)



"""
        outcome = mean(tn[other])
        product of all trustworthiness scores for a given question for all cardinalities, summed for all questions that the evaluated tn is involved in

        """

        # y = np.log((math.gamma(1+t_n))/(math.gamma(1)*math.gamma(t_n)))+(t_n-1)*np.sum([
        #     np.log(1-(
        #         (t_n if k == n[1][f"q_{m}"] else ((1 - t_n) / self.cm)
        #          )
        #     ))])

        # y = np.prod([
        #         np.prod([
        #             np.prod([(t_n if k == n[1][f"q_{m}"] else ((1 - t_n) / self.cm))
        #             for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #         ])
        #         for k in range(self.K)])
        # for m in annotations.loc[(annotations['id_0']==n[1].ID)|
        #                          (annotations['id_1']==n[1].ID)|
        #                          (annotations['id_2']==n[1].ID)].index])


        # attempt with basically H = gamma(tn)/gamme(tn_new)
        # y= np.sum([
        #     np.sum([
        #          np.prod([
        #             (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"]) / self.cm)) * (1 / self.K) * n[1][
        #                 "T_model"]
        #             for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #          ])/sum([
        #             np.prod([
        #                 (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"]) / self.cm)) * (1 / self.K) * n[1][
        #                     "T_model"]
        #                 for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #             ]
        #             ) for l in self.L
        #          ])
        #         for k in range(self.K)
        #     ])
        #     for m in annotations.loc[(annotations['id_0'] == n[1].ID) |
        #                              (annotations['id_1']==n[1].ID)|
        #                              (annotations['id_2']==n[1].ID)].index
        # ])


        # y = sum([
        #         sum([gamma.iloc[m, k] * (np.log(t_n) if k == n[1][f"q_{m}"] else (np.log(1 - t_n)/ self.cm))
        #
        #              for k in range(self.K)])
        #
        #         for m in self.M])

        # y = sum([n[1][f"q_{m}"]
        #          )
#
    # def gamma(self, k, user, m):
    #     """
    #     probability of true label for question m being equal to option k given the provided labels and the trustworthiness
    #     """
    #
    #
    #     # product of the modelled trustworthiness if answer is equal to k iterated over every user who answered question m
    #     # sum of all products of the modelled trustworthiness if answer is equal to answer option l iterated over every user who answered question m over all options l
    #
    #     """
    #     p(l_nm | GT_m == k, t_n) is implemented as (t_n if k==l_nm else (1-t_n)/cm)
    #     p(GT_m == k) and p(GT_m == l) implemented as (1/k)
    #     """
    #
    #     num = np.prod([
    #                     (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * n[1]["T_model"]
    #                    for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
    #                 ])
    #     denom = sum([
    #                 np.prod([
    #                         (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) * n[1]["T_model"]
    #                           for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
    #                 ]
    #                 ) for l in self.L])
    #     g = num/denom
    #     return g

    # @timeit
    # def e_step(self):
    #     for k in range(self.K):  # for each option
    #         with Pool(16) as p:
    #             result = p.map(partial(self.gamma,k, user), self.M)
    #         self.gamma_.loc[:,k] = result
    #     return self.gamma_