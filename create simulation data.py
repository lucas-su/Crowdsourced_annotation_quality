import math
import pickle
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas
import  random

def dist_annot(users, annotations, q_id):
    if q_id % 100 == 0:
        print(q_id)

    userlist = np.zeros(duplication_factor)

    while not (len(set(userlist)) == len(userlist)):
        userlist = [random.randint(0,users.__len__()-1) for _ in range(duplication_factor)]
    answers = [sim_answer(users, annotations, u_id, car, q_id) for u_id in userlist]

    return  userlist, answers

def sim_answer(users, annotations, u_id, car, q_id):
    if users.loc[u_id].type == "first_only" and mode[:6] !="single":
        ans = 0
    else:
        ans = annotations.loc[q_id,"GT"] if users.loc[users.ID==u_id].T_given.values.item() > (random.random()) else random.randint(0,car-1)
    return ans

class dist():
    def __init__(self, param, x):
        self.param = param
        self.x = x
        self.cum = self.build()

    def __call__(self, *args, **kwargs):
        return self.cum

    def build(self):
        cum = np.zeros(self.x.__len__())
        for dist in self.param:
            func = eval(f'self.{dist[0]}') # eval to determine which distribution to take. Options: gamma beta gaussian uniform
            if func.__name__ not in ['gamma', 'beta', 'gaussian', 'uniform', 'single']:
                raise NotImplementedError
            cum += func(*dist[1:])

        return cum/self.param.__len__()

    def single(self, prob):
        if prob == 1:
            return np.append(np.zeros(self.x.shape[0]-1),prob)
        elif prob == 0:
            return np.append(1-prob, np.zeros(self.x.shape[0] - 1))

    def gamma(self, alpha, beta):
        # alpha = 1
        # beta = 1
        return 1/(beta**(alpha)*math.gamma(alpha))*self.x**(alpha-1)*math.e**(-self.x/beta)

    def beta(self, alpha, beta):
        # alpha = 13
        # beta = 13
        y = (1/max(self.x))*self.x # normalize to entire space instead of 0,1
        return ((math.gamma(alpha+beta))/(math.gamma(alpha)*math.gamma(beta)))*(y**(alpha-1))*(1-y)**(beta-1)

    def gaussian(self, mu, sd):
        return (math.e ** (-((self.x - mu) ** 2) / 2 * (sd) ** (2))) / sd * math.sqrt(2 * math.pi)

    def uniform(self):
        return self.x * (max(self.x) / self.x.__len__())

    def plot(self):
        fig, ax = plt.subplots(2,1)

        # ax[0].plot(x, gam_beta_mean, linewidth=2.0)
        ax[0].plot(x, self.cum, linewidth=2.0)

        # ax[1].plot(x, gam_y, linewidth=2.0)
        # ax[1].plot(x, gauss_y, linewidth=2.0)
        # ax[1].plot(x, beta_y, linewidth=2.0)

        ax[0].set(xlim=(0, xmax), xticks=np.arange(1, xmax),
               ylim=(0, 1), yticks=np.arange(0.1, 1, 0.1))
        ax[1].set(xlim=(0, xmax), xticks=np.arange(1, xmax),
               ylim=(0, 1), yticks=np.arange(0.1, 1, 0.1))
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,10,self.cum.__len__()), self.cum, 'o')
        plt.show()


if __name__ == "__main__":

    nAnnot = 50
    nQuestions = 200
    car = 10
    duplication_factor = 5
    p_first_only = 0.1

    # other globals
    xmax = 1
    steps = 1000
    x = np.linspace(0, xmax, steps)

    ##################
    # 50 50 gaussian #
    ##################
    # mode = "50_50_gaussian"
    # param = [["gaussian",2,1],["gaussian",8,1]]
    # distribution = dist(param, x)

    ###########
    # uniform #
    ###########

    mode = "uniform"
    param = [['uniform']]
    distribution = dist(param, x)

    ###########################
    # single trustworthiness #
    ###########################

    # mode = "single0"
    # param = [['single', 0]]
    # distribution = dist(param, x)
    #
    # mode = "single1"
    # param = [['single', 1]]
    # distribution = dist(param, x)

    # datasets
    udata = {"ID":range(nAnnot),
             "type": ["normal" if random.random()>p_first_only else "first_only" for _ in range(nAnnot)],
             "T_given": random.choices(x, distribution(), k=nAnnot),
             "T_model": np.ones(nAnnot)*0.5}

    for q in range(nQuestions): # keep track of labels in broad format
        udata[f"q_{q}"] = np.ones(nAnnot)*np.nan

    user = pandas.DataFrame(data=udata)

    annotdict = {"GT": random.choices(range(car), k=nQuestions),
                 "model": np.zeros(nQuestions)}
    for i in range(duplication_factor):
        annotdict[f'id_{i}'] = np.zeros(nQuestions)
        annotdict[f'annot_{i}'] = np.zeros(nQuestions)

    annotation = pandas.DataFrame(data=annotdict)


    with Pool(16) as p:
        results = p.map(partial(dist_annot, user, annotation), range(nQuestions))


    res = np.array([np.concatenate(np.column_stack(i)) for i in results])

    annotation.loc[:, np.concatenate([[f'id_{i}',f'annot_{i}'] for i in range(duplication_factor)])] = res

    for i, q in enumerate(results):
        for u_a_pair in zip(q[0],q[1]):
            user.loc[u_a_pair[0], f'q_{i}'] = u_a_pair[1]

    print("done, saving")

    # save data
    with open(f'simulation data/{mode}_dup-{duplication_factor}_car-{car}_p-fo-{p_first_only}_user.pickle', 'wb') as file:
        pickle.dump(user, file)
    with open(f'simulation data/{mode}_dup-{duplication_factor}_car-{car}_p-fo-{p_first_only}_annotations_empty.pickle', 'wb') as file:
        pickle.dump(annotation, file)