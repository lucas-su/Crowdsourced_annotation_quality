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

import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
import pprint


class EM():


    def __init__(self, K):
        self.N = np.arange(0,user.__len__()) # annotators
        self.M = np.arange(0,nQuestions) # questions
        self.L = np.arange(0,K) # given label per question
        self.K = K
        self.cm = K-1 # -1 because there's one good answer and the rest is wrong
        self.gamma_ = pandas.DataFrame(columns=[i for i in range(K)])

    def to_cat(y, num_classes=None, dtype=float):
        """
        Helpen function to transform to categorical format
        :param num_classes:
        :param dtype:
        :return:
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def gamma_vectorized(self, m, k):
        """
        responsibilities implemented in vectorized manner to speed up processing, not done yet
        :param m:
        :param k:
        :return:
        """
        pass
        # n_Am_k = to_cat(user.loc[:,[f'q_{i}' for i in range(k)]]) # select question answered matrix and make categorical tensor from user table
        # n_am_k_inv = 1- n_Am_k
        #
        # tn_k = np.tile(user.loc[:,"t_model"],k)
        # tn_k_inv = (1-tn_k)/self.cm
        #
        # term1 = np.multiply(n_Am_k, tn_k)
        # term2 = np.multiply(n_am_k_inv, tn_k_inv)
        #
        # factor1 = np.add(term1,term2) # n x m x k
        # (factor1 / self.cm * (1 / self.K))
        #
        # denom = sum(np.prod((factor1 / self.cm * (1 / self.K)), axis=0).__array__(), axis=1) # vector of k?
        #
        # num =


    def gamma(self, m, k):
        """
        probability of true label for question m being equal to option k given the provided labels and the trustworthiness
        :param m:
        :param k:
        :return:
        """


        # product of the modelled trustworthiness if answer is equal to k iterated over every user who answered question m
        # sum of all products of the modelled trustworthiness if answer is equal to answer option l iterated over every user who answered question m over all options l

        """
        p(l_nm | GT_m == k, t_n) is implemented as (t_n if k==l_nm else (1-t_n)/cm)
        p(GT_m == k) and p(GT_m == l) implemented as (1/k)
        """

        num = np.prod([
                        (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * n[1]["T_model"]
                       for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ])
        denom = sum([
                    np.prod([
                            (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) * n[1]["T_model"]
                              for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ]
                    ) for l in self.L])
        g = num/denom
        return g


    def e_step(self):
        for m in self.M: # for each question
            for k in range(self.K): # for each option
                self.gamma_.loc[m,k] = self.gamma(m, k)
        return self.gamma_

    def m_step(self, gamma, nq, n):
        n = n[1] # with multiprocessing setup, n is a tuple of (ID, Series) and we only need the Series here
        # construct list of answered questions for current annotators
        l_n = []
        for i in range(nq):
            if ~np.isnan(n[f"q_{i}"]):
                l_n.append(i)

        nom = sum([
                sum([
                    (gamma.loc[m,k] if k == n[f"q_{m}"] else 0)
                for k in list(range(car))])
              for m in l_n
        ])

        denom = sum([sum([gamma.loc[m,k] for k in range(car)])
            for m in l_n])

        return nom/denom

def run_em(iterations, car, nQuestions):
    ems.loc[(ems['iterations'].values == iterations) &
            (ems['car'].values == car) &
            (ems['mode'].values == mode) &
            (ems['dup'].values == dup) &
            (ems['p_fo'].values == p_fo), 'EM'] = EM(car)
    i = 0
    while i < iterations:
        print("iteration: ", i)
        g = ems.loc[(ems['iterations'].values == iterations) &
            (ems['car'].values == car) &
            (ems['mode'].values == mode) &
            (ems['dup'].values == dup) &
            (ems['p_fo'].values == p_fo), 'EM'].values[0].e_step()
        with Pool(16) as p:
            results = p.map(partial(ems.loc[(ems['iterations'].values == iterations) &
            (ems['car'].values == car) &
            (ems['mode'].values == mode) &
            (ems['dup'].values == dup) &
            (ems['p_fo'].values == p_fo), 'EM'].values[0].m_step, g, nQuestions), user.iterrows())
        # for ann in user.ID:
        #     user.loc[ann, f"t_weight_{i}"] = m.step(g, user.iloc[ann], nQuestions)
        user.loc[:, "T_model"] = results
        user.loc[:, f"t_weight_{i}"] = results
        i += 1
    for q in range(nQuestions):
        k_w = []
        for k in range(car):
            d_w = 0
            for d in range(dup):
                if annotations.loc[q, f'annot_{d}'] == k:
                    d_w += user.loc[annotations.loc[q, f'id_{d}'], 'T_model']
            k_w.append(d_w)
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
    ems.loc[(ems['iterations'].values == iterations) &
            (ems['car'].values == car) &
            (ems['mode'].values == mode) &
            (ems['dup'].values == dup) &
            (ems['p_fo'].values == p_fo), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
    ems.loc[(ems['iterations'].values == iterations) &
            (ems['car'].values == car) &
            (ems['mode'].values == mode) &
            (ems['dup'].values == dup) &
            (ems['p_fo'].values == p_fo), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))
    summary = {"Mode": mode,
               "Cardinality": car,
               "Iterations": iterations,
               "Duplication factor": dup,
               "Proportion 'first only'": p_fo,
               "Percentage correct modelled": 100 * (1 - (diff_m_cnt / nQuestions)),
               "Percentage correct naive": 100 * (1 - (diff_n_cnt / nQuestions))}
    [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":

    iterations_list = [5,10,15,20]
    car_list = list(range(2,10))
    modes = ['gaussian']
    dups = [3,5,7,9]
    p_fos = [0.0,0.1,0.2,0.3]


    # iterations = 2     # iterations of EM algo
    # car = 5
    # mode = "uniform"    # data modes, options: real, single0 (perfectly bad trustworthiness), single1 (perfect trustworthiness), uniform, gaussian (all except real are simulated)
    # dup = 3             # duplication factor, determines which premade simulation dataset to use
    # p_fo = 0.0          # proportion 'first only' annotators, who are lazy and only ever click the first option
    ###############################



    ems = pandas.DataFrame(columns=['iterations', 'car', 'mode', 'dup', 'p_fo', 'EM', 'pc_m', 'pc_n'])
    for iterations in iterations_list:
        for car in car_list:
            for mode in modes:
                for dup in dups:
                    for p_fo in p_fos:
                        if mode == 'real':
                            with open('data/users.csv', 'r') as file:
                                user = pandas.read_csv(file)

                            with open('data/web_annotations_colnamens.pylist', 'r') as file:
                                colnames = eval(file.read())

                            with open('data/web_annotations.csv', 'r') as file:
                                annotation = pandas.read_csv(file, names=colnames)
                        else:
                            with open(f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_user.pickle',
                                      'rb') as file:
                                user = pickle.load(file)
                            with open(
                                    f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_annotations_empty.pickle',
                                    'rb') as file:
                                annotations = pickle.load(file)
                            # car = annotations.loc[:,np.concatenate([[f'annot_{i}'] for i in range(dup)])].values.max()+1
                            # init user weights at 1
                        for i in range(iterations + 1):
                            user[f't_weight_{i}'] = np.ones(
                                user.__len__()) * 0.5  # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time
                        user['included'] = np.ones(user.__len__())

                        # nAnnot = user.__len__()
                        nQuestions = annotations.__len__()
                        ems.loc[ems.__len__(), :] = [iterations, car, mode, dup, p_fo, None, 0, 0]
                        run_em(iterations, car, nQuestions)
    with open(f'data/em_data_{"_".join(modes)}.pickle', 'wb') as file:
        pickle.dump(ems, file)
