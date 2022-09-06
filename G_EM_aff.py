import csv

import numpy
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

    def gamma(self, k, user, m):
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
        if list(user.loc[[type(i)== np.ndarray for i in user.loc[:,f"q_{m}"]]].iterrows()) == []:
            return 0

        else:

            num = np.prod([
                            (n[1]["T_model"] if n[1][f"q_{m}"][k] == 1 else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * n[1]["T_model"]
                           for n in user.loc[[type(i)==np.ndarray for i in user.loc[:,f"q_{m}"]]].iterrows()# user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows() # user.loc[[type(i)=='object' for i in user.loc[:,f"q_{m}"]]         ].iterrows()
                        ])
            denom = sum([
                        np.prod([
                                (n[1]["T_model"] if n[1][f"q_{m}"][l] == 1 else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) * n[1]["T_model"]
                                  for n in user.loc[[type(i)==np.ndarray for i in user.loc[:,f"q_{m}"]]].iterrows()
                        ]
                        ) for l in self.L])
            if denom == 0:
                return np.spacing(1) # if users who have exactly 0 as their T are the only ones who answer a given question, return very small value for g
            else:
                g = num/denom
                return g


    def e_step(self):
        # for m in self.M: # for each question
        #     for k in range(self.K): # for each option
        #         self.gamma_.loc[m,k] = self.gamma(k, m)
        for k in range(self.K):  # for each option
            with Pool(16) as p:
                result = p.map(partial(self.gamma,k, user), self.M)
            self.gamma_.loc[:,k] = result
        return self.gamma_



    def m_step(self, gamma, nq, car, n):
        n = n[1] # with multiprocessing setup, n is a tuple of (ID, Series) and we only need the Series here
        # construct list of answered questions for current annotators
        l_n = []
        for i in range(nq):
            if type(n[f"q_{i}"]) == np.ndarray:
                l_n.append(i)
        if l_n == []:
            return 0
        else:
            nom = sum([
                    sum([
                        (gamma.loc[m,k] if n[f"q_{m}"][k] == 1 else 0)
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
            (ems['p_fo'].values == p_fo), 'EM'].values[0].m_step, g, nQuestions, car), user.iterrows())

        user.loc[:, "T_model"] = results
        user.loc[:, f"t_weight_{i}"] = results
        i += 1

    # annotations.insert(annotations.columns.get_loc("object_label") + 1, "model", np.zeros(nQuestions))
    # annotations['model'] = annotations['model'].astype('object')
    model = pandas.Series(np.zeros(nQuestions), dtype='object')
    naive = pandas.Series(np.zeros(nQuestions), dtype='object')
    for q in range(nQuestions):
        k_w = np.zeros(car)
        for k in range(car):
            # d_w = 0
            dobreak = False
            n_ann = 0
            for d in range(dup):
                u = annotations.loc[q, f'anno_{d+1}_id']-1
                if np.isnan(u):
                    dobreak = True
                    break
                else:
                    n_ann += 1
                if user.loc[u, f'q_{q}'][k] == 1:
                    # k_w[k] += user.loc[annotations.loc[q, f'id_{d}'], 'T_model']
                    k_w[k]  += user.loc[u, 'T_model']
                    # k_w = [k_w[i] + ((1 - user.loc[u, 'T_model']) / (car - 1)) if i != k else
                    #     k_w[i] + user.loc[u, 'T_model'] for i in range(car)]
                else:
                    k_w[k] += 1-user.loc[u, 'T_model']
                    # k_w = [k_w[i] + (( user.loc[u, 'T_model']) / (car - 1)) if i != k else
                    #        k_w[i] + 1-user.loc[u, 'T_model'] for i in range(car)]
            if dobreak:
                break

        if n_ann==0:
            # annotations.loc[q, 'model'] = np.nan
            model.loc[q] = np.nan
        else:
            val = (pandas.Series([1 if (kwi/(n_ann)) > 0.5 else 0 for kwi in k_w]))
            # annotations.loc[q, 2:] = annotations.loc[q, :].astype('object')
            # annotations.loc[q, 'model'] = np.array(val).astype('object')
            model.loc[q] = np.array(val).astype('object')


    for q in range(nQuestions):
        k_w = []
        for k in range(car):
            d_w = 0
            dobreak = False
            n_ann = 0
            for d in range(dup):
                u = annotations.loc[q, f'anno_{d + 1}_id']-1
                if np.isnan(u):
                    dobreak = True
                    break
                else:
                    n_ann += 1
                if user.loc[u, f'q_{q}'][k] == 1:
                    d_w += 1.1 # in a tie of 1-0, 1 should be kept
                else:
                    d_w -= 1

            k_w.append(d_w)

        if n_ann == 0:
            # annotations.loc[q, 'naive'] = np.nan
            naive.loc[q] = np.nan
        else:
            val = (pandas.Series([1 if kwi > 0 else 0 for kwi in k_w]))
            naive.loc[q] = np.array(val).astype('object')
            # annotations.loc[q, 'naive'] = np.array([1 if kwi > 0 else 0 for kwi in k_w]).astype('object')

    diff = []
    for i, (n, m) in enumerate(zip(naive.loc[naive.notnull()],model.loc[model.notnull()])):
        if not np.array_equal(n,m):
            diff.append(i)
    summary = {"Mode": mode,
               "Cardinality": car,
               "Iterations": iterations,
               'diff': diff,
               'naive': naive.loc[naive.notnull()],
               'model': model.loc[model.notnull()]
               }
    with open(f'data/aff_naive_{level}.csv', 'w') as file:
        cw = csv.writer(file)
        cw.writerows(naive.loc[naive.notnull()])
    with open(f'data/aff_model_{level}.csv', 'w') as file:
        cw = csv.writer(file)
        cw.writerows(model.loc[model.notnull()])
    [print(f'{key:<30} {summary[key]}') for key in summary.keys()]



if __name__ == "__main__":

    iterations_list = [2]



    # iterations = 2     # iterations of EM algo
    # car = 5
    # mode = "uniform"    # data modes, options: real, single0 (perfectly bad trustworthiness), single1 (perfect trustworthiness), uniform, gaussian (all except real are simulated)
    # dup = 3             # duplication factor, determines which premade simulation dataset to use
    # p_fo = 0.0          # proportion 'first only' annotators, who are lazy and only ever click the first option
    ###############################


    ems = pandas.DataFrame(columns=['iterations', 'car', 'mode', 'dup', 'p_fo', 'EM', 'pc_m', 'pc_n'])
    level = 'all'
    dup = 3
    p_fo = 0
    mode = 'real'
    all_affs = ["con_move", "uncon_move", "dir_affs", "indir_affs", "observe_affs", "social_affs", "no_affs",
                   "no_clue", "roll", "push", "drag", "tether", "pick_up_carry", "pour", "fragile", "open", "grasp",
                   "pull", "tip", "stack", "cut_scoop", "support", "transfer", "requires_other", "info", "deco",
                   "together", "none", "warmth", "illumination", "walk"]
    if level == 'high':
        car = 7 # 7 for high level affs, 23 for low level
        affs = all_affs[:7]
    elif level == 'low':
        car = 23
        affs = all_affs[8:]
    elif level == 'high_low':
        car = 30
        affs = all_affs[:7] + all_affs[8:]
    else:
        car = 31
        affs = all_affs


    with open('D:\\sunrgbd\\users.csv', 'r') as file:
        user = pandas.read_csv(file, header=None)

    with open('D:\\sunrgbd\\colnames.pylist', 'r') as file:
        colnames = eval(file.read())

    with open('D:\\sunrgbd\\web_annotations updated.csv', 'r') as file:
        annotations = pandas.read_csv(file, names=colnames)
    user[0] = user[0]-1 # sql export starts count at 1


    qs = pandas.DataFrame(np.zeros((user.__len__(), annotations.__len__())), columns=[f'q_{i}' for i in range(annotations.__len__())])
    user = pandas.concat((user, qs), axis=1)
    user = user.astype('object')
    for i, row in enumerate(annotations.iterrows()):
        for annot in [1,2,3]:
            if ~np.isnan(row[1][f'anno_{annot}_id']):
                val = ([row[1].loc[f'anno_{annot}_{j}'] for j in affs ])
                user.loc[row[1][f'anno_{annot}_id']-1, :] = user.loc[row[1][f'anno_{annot}_id']-1, :].astype('object')
                user.loc[row[1][f'anno_{annot}_id']-1, f'q_{i}'] = np.array(val).astype('object')

    for iterations in iterations_list:
        for i in range(iterations + 1):
            user[f't_weight_{i}'] = np.ones(
                user.__len__()) * 0.5  # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time
        user['included'] = np.ones(user.__len__())
        user.insert(2, 'T_model', 0.5*np.ones(user.__len__())) # users start at modelled T of 0.5
        # nAnnot = user.__len__()
        nQuestions = annotations.__len__()
        ems.loc[ems.__len__(), :] = [iterations, car, mode, dup, p_fo, None, 0, 0]
        run_em(iterations, car, nQuestions)

        with open(f'data/user_data_{mode}_{level}.pickle', 'wb') as file:
            pickle.dump(user, file)
    with open(f'data/em_data_{mode}_{level}.pickle', 'wb') as file:
        pickle.dump(ems, file)
