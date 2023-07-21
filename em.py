import os
import platform
from settings import *
from datetime import datetime

import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial

from create_simulation_data import createData
from majority_vote import majority

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class EM():
    def __init__(self, K):
        self.N = users['ID'] # annotators
        self.M = np.arange(0,nQuestions) # questions
        self.L = np.arange(0,K) # given label per question
        self.K = K
        self.cm = K-1 # -1 because there's one good answer and the rest is wrong
        self.gamma_ = pandas.DataFrame(columns=[i for i in range(K)])

        self.annotators = {id: Annotator(id, type, T_given, car) for id, type, T_given in zip(users.ID, users.type, users.T_given)}
        self.questions = {qid: Question(qid, KG, GT, car) for qid, KG, GT in zip(items.ID, items.KG, items.GT)}

        self.annotations = []

        for row in items.iterrows():
            for i in range(dup):
                self.annotations.append(Annotation(self.annotators[row[1][f'id_{i}']], self.questions[row[0]], row[1][f'annot_{i}']))


    # def gamma(self, k, users, items, m):
    #     """
    #     probability of true label for question m being equal to option k given the provided labels and the trustworthiness
    #     :param m:
    #     :param k:
    #     :return:
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
    #     if items.loc[m,'KG'] == True:
    #         if items.loc[m, 'GT'] == k:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #         num = np.prod([
    #                         (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) #* beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
    #                        for n in users.loc[~np.isnan(users.loc[:,f"q_{m}"])].iterrows()
    #                     ])
    #         denom = sum([
    #                     np.prod([
    #                             (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) # * beta.pdf(n[1]["T_model"], n[1]['a'], n[1]['b'])
    #                               for n in users.loc[~np.isnan(users.loc[:,f"q_{m}"])].iterrows()
    #                     ]
    #                     ) for l in self.L])
    #         g = num/denom
    #         return g
    #
    #
    # def e_step(self):
    #     # for m in self.M: # for each question
    #     #     for k in range(self.K): # for each option
    #     #         self.gamma_.loc[m,k] = self.gamma(k, m)
    #     for i in range(self.K):
    #         self.gamma_.loc[:, i] = np.zeros(items.__len__())
    #     kgm = items.loc[(items['KG']==True), 'ID']
    #     for k in range(self.K):  # for each option
    #         with Pool(32) as p:
    #             result = p.map(partial(self.gamma,k, users, items), kgm)
    #         self.gamma_.loc[(items['KG']==True),k] = result
    #
    #     m = items.loc[(items['KG'] == False), 'ID']
    #     for k in range(self.K):  # for each option
    #         with Pool(32) as p:
    #             result = p.map(partial(self.gamma, k, users, items), m)
    #         self.gamma_.loc[(items['KG']==False), k] = result
    #
    #     return self.gamma_
    #
    #
    # def m_step(self, gamma, nq, car, n):
    #     n = n[1] # with multiprocessing setup, n is a tuple of (ID, Series) and we only need the Series here
    #     # construct list of answered questions for current annotators
    #     l_n = []
    #     for i in range(nq):
    #         if ~np.isnan(n[f"q_{i}"]):
    #             l_n.append(i)
    #
    #     nom = sum([
    #             sum([
    #                 (gamma.loc[m,k] if k == n[f"q_{m}"] else 0)
    #             for k in list(range(car))])
    #           for m in l_n
    #     ])
    #
    #     denom = sum([sum([gamma.loc[m,k] for k in range(car)])
    #         for m in l_n])
    #
    #     return nom/denom

class Question():
    def __init__(self, id, KG, GT, car, difficulty=None):
        self.id = id
        # self.prior = np.array([priors['qAlpha'] for _ in range(car)])

        # self.posterior = np.array([priors['qAlpha'] for _ in range(car)])
        # self.basePrior = np.array([priors['qAlpha'] for _ in range(car)])
        # self.cardinality = len(self.prior)
        self.KG = KG
        self.GT = GT
        self.C = 0  # start at 0 to prevent sampling from annealing in first iteration
        self.car = car
        self.diff = difficulty
        self.annotations = []  # Keep track of the labels that were made for this question
        self.postsamples = []
        self.model = np.random.choice(range(car))
        self.w = np.random.random() # weight
    def addAnnotation(self, annot):
        self.annotations.append(annot)

    def e_step(self):
        a = np.sum([(annot.annotator.sensitivity**(annot.value))*((1-annot.annotator.sensitivity)**(1-annot.value)) for annot in self.annotations])
        b = np.sum([(annot.annotator.sensitivity ** (1-annot.value)) * ((1 - annot.annotator.sensitivity) ** (annot.value)) for annot in self.annotations])
        p = sigmoid(self.w)
        self.model = (a*p)/(a*p+b*(1-p))


class Annotator():
    def __init__(self, id, type, T_given, car):
        self.id = id
        self.T = T_given
        self.KG = True if type == 'KG' else False
        self.annotations = []
        # self.basePrior = np.array((priors['aAlpha'], priors['aBeta']))
        # self.prior = np.array((priors['aAlpha'], priors['aBeta']))
        # self.posterior = np.array((priors['aAlpha'], priors['aBeta']))
        self.sensitivity = 0
        self.specificity = 0

        # self.postsamples = []
        self.C = 0  # start at 0 to prevent sampling from annealing in first iter
        self.car = car
        # self.annealdist = np.array(priors['anneal'])

    def addAnnotation(self, annot):
        self.annotations.append(annot)

    def m_step(self):
        self.sensitivity = (np.sum([(annot.value == annot.question.GT) for annot in self.annotations]))/ np.sum([annot.value for annot in self.annotations])
        self.specificity = (np.sum([(1-annot.value == 1-annot.question.GT) for annot in self.annotations]))/ np.sum([1-annot.value for annot in self.annotations])


class Annotation:
    def __init__(self, annotator, question, value):
        self.annotator = annotator
        annotator.addAnnotation(self)
        self.question = question
        question.addAnnotation(self)
        self.value = value
        # self.value1ofk = np.eye(question.prior.__len__())[value]
        # debug("Annotation constructor",self.value,self.value1ofk)


def e_multi(question):
    question.e_step()

def m_multi(annotator):
    annotator.m_step()


def run_em(iterations, car, nQuestions):
    models = []
    n=0
    eta = 0.1
    while n < nModels:
        models.append(EM(car))
        model = models[-1]

        i = 0
        while i < iterations:
            print("iteration: ", i)
            # e step
            with Pool(32) as p:
                 p.map(e_multi, model.questions.values())

            # m step for the rest
            with Pool(32) as p:
                nonKGAnnots = [annotator for annotator in model.annotators.values() if annotator.KG != True]
                p.map(m_multi,nonKGAnnots)


            for q in model.questions.values():
                g = q.model - sigmoid(q.w)
                H = sigmoid(q.w) * sigmoid(1 - q.w)

                q.w = q.w - eta*(1/H)*g
                print(sigmoid(q.w))
                print(q.GT)
            print(sum([round(sigmoid(q.w))==q.GT for q in model.questions.values()]))
            print(model.questions.__len__())

            i += 1


        for q in range(nQuestions):
            k_w = np.zeros(car)
            for k in range(car):

                for d in range(dup):
                    if items.loc[q, f'annot_{d}'] == k:
                        k_w = [k_w[i] + ((1 - users.loc[items.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                               k_w[i] + users.loc[items.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
                        # k_w[k] += users.loc[items.loc[q, f'id_{d}'], 'T_model']
                    else:
                        # k_w = [k_w[i]+((1-users.loc[items.loc[q, f'id_{d}'], 'T_model'])/(car-1)) if i!= k else k_w[i] for i in range(car)]
                        k_w = [
                            k_w[i] + ((users.loc[items.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                            k_w[i] + 1-users.loc[items.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
            items.loc[q, 'model'] = k_w.index(max(k_w))
        items.insert(items.columns.get_loc("model") + 1, "naive", np.zeros(nQuestions))
        for q in range(nQuestions):
            k_w = []
            for k in range(car):
                d_w = 0
                for d in range(dup):
                    if items.loc[q, f'annot_{d}'] == k:
                        d_w += 1
                k_w.append(d_w)
            items.loc[q, 'naive'] = k_w.index(max(k_w))

        diff_m = items.loc[:, 'GT'] - items.loc[:, 'model']
        diff_n = items.loc[:, 'GT'] - items.loc[:, 'naive']
        diff_m_cnt = (diff_m != 0).sum()
        diff_n_cnt = (diff_n != 0).sum()
        em_data.loc[(em_data['size'].values == size) &
                    (em_data['iterations'].values == iterations) &
                    (em_data['car'].values == car) &
                    (em_data['mode'].values == T_dist) &
                    (em_data['dup'].values == dup) &
                    (em_data['p_fo'].values == p_fo) &
                    (em_data['kg_q'].values == kg_q) &
                    (em_data['kg_u'].values == kg_u), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
        em_data.loc[(em_data['size'].values == size) &
                    (em_data['iterations'].values == iterations) &
                    (em_data['car'].values == car) &
                    (em_data['mode'].values == T_dist) &
                    (em_data['dup'].values == dup) &
                    (em_data['p_fo'].values == p_fo) &
                    (em_data['kg_q'].values == kg_q) &
                    (em_data['kg_u'].values == kg_u), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))
        summary = {"Mode": T_dist,
                   "Cardinality": car,
                   "Iterations": iterations,
                   "Duplication factor": dup,
                   "Proportion 'first only'": p_fo,
                   "Proportion 'known good'": kg_q,
                   "Percentage correct modelled": 100 * (1 - (diff_m_cnt / nQuestions)),
                   "Percentage correct naive": 100 * (1 - (diff_n_cnt / nQuestions))}
        [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":

    iterations_list = [10]

    T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
    dup_list = [3]
    p_fo_list = [0.0, 0.1]
    kg_q_list = [0.0, 0.1]
    kg_u_list = [0.0, 0.1]

    session_folder = f'session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    os.makedirs(f'{os.getcwd()}/sessions/car{car_list[0]}/{session_folder}/output', exist_ok=True)

    # if not platform.system() == 'Windows':



    em_data = pandas.DataFrame(columns=['size', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'kg_q', 'kg_u', 'EM', 'pc_m', 'pc_n'])
    # for size in ['small', 'medium', 'large']:
    #     for iterations in iterations_list:
    #         for car in car_list:
    #             for T_dist in T_dist_list:
    #                 for dup in dup_list:
    #                     for p_fo in p_fo_list:
    #                         for kg_q in kg_q_list:
    #                             for kg_u in kg_u_list:
    for size in datasetsize_list:
        for sweeptype, T_dist_list in sweeps.items():
            for car in car_list:
                for dup in dup_list:
                    priors = set_priors()
                    for p_fo in p_fo_list:
                        for kg_q in kg_q_list:
                            for kg_u in kg_u_list:
                                em_data = pandas.DataFrame(
                                    columns=['size', 'iterations', 'car', 'T_dist', 'sweeptype', 'dup', 'p_fo', 'kg_q',
                                             'kg_u', 'em', 'pc_m', 'pc_n', 'pc_n_KG', 'CertaintyQ', 'CertaintyA'])

                                session_dir = set_session_dir(size, sweeptype, car, dup, p_fo, kg_q,
                                                              kg_u) + f'session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
                                for T_dist in T_dist_list:
                                    os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)

                                    createData(f'{session_dir}', car, T_dist, dup, p_fo, kg_u, ncpu, size)
                                    print(f"Datasetsize {size}, cardinality {car}, distribution {T_dist}, annotations per item {dup}, malicious {p_fo}, known good items {kg_q}, known good users {kg_u}")

                                    with open(
                                            f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_user.pickle',
                                            'rb') as file:
                                        users = pickle.load(file)
                                    with open(
                                            f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_annotations_empty.pickle',
                                            'rb') as file:
                                        items = pickle.load(file)

                                    ## add parameters to dataframes
                                    # known goods
                                    items[f'KG'] = np.zeros(items.__len__())
                                    if kg_q > 0:
                                        items.loc[:kg_q - 1, f'KG'] = 1


                                    # global nQuestions
                                    nQuestions = items.__len__()
                                    # majority(items, nQuestions, car)
                                    maj = majority(items, nQuestions, car, dup, users)

                                    # user['included'] = np.ones(user.__len__())
                                    run_em(emIters, car, nQuestions)



                                    # init user weights at 1
                                    # for i in range(iterations + 1):
                                    #     user[f't_weight_{i}'] = np.ones(user.__len__()) * 0.5  # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time

                                    # save data
                                    with open(f'{session_dir}/output/em_annotations_T_dist-{T_dist}_iters-{emIters}.pickle', 'wb') as file:
                                        pickle.dump(items, file)
                                    with open(f'{session_dir}/output/em_user_T_dist-{T_dist}_iters-{emIters}.pickle', 'wb') as file:
                                        pickle.dump(users, file)
                                    with open(f'{session_dir}/output/em_data.pickle', 'wb') as file:
                                        pickle.dump(em_data, file)
