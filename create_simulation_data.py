import math
import os
import pickle
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas
import  random

# set globals
xmax = 1
steps = 1000
x = np.linspace(0, xmax, steps)


def dist_annot(users, annotations, dup, car, mode, q_id):
    userlist = np.zeros(dup)

    while not (len(set(userlist)) == len(userlist)):
        userlist = [random.randint(0,users.__len__()-1) for _ in range(dup)]
    answers = [sim_answer(users, annotations, u_id, car, q_id, mode) for u_id in userlist]

    return  userlist, answers

def sim_answer(users, annotations, u_id, car, q_id, mode):
    if users.loc[u_id].type == "first_only" and mode[:6] !="single":
        ans = 0
    elif users.loc[u_id].type == "KG" and mode != "single0":
        ans = annotations.loc[q_id,"GT"]
    else:
        ans = annotations.loc[q_id,"GT"] if users.loc[users.ID==u_id].T_given.values.item() > (random.random()) else \
            random.choice(list(set(np.arange(0,car)) - {annotations.loc[q_id, "GT"]}))
 #random.randint(0,car-1) # use randint if 0 trustworthiness means chance level
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
    #
    # def beta(self, a,b):
    #     (((self.x)**(a-1))*((1-self.x)**(b-1)))/self.B(a,b)
    #
    # def B(self, a,b):
    #     return

    def single(self, prob):
        probs = np.zeros(self.x.shape[0])
        if float(prob) == 1.:
            probs[-1] = 1
        else:
            probs[int(float(prob) * self.x.shape[0])] = 1
        return probs

    def gamma(self, alpha, beta):
        # alpha = 1
        # beta = 1
        return 1/(beta**(alpha)*math.gamma(alpha))*self.x**(alpha-1)*math.e**(-self.x/beta)

    def beta(self, a,b):
        return (((self.x) ** (a - 1)) * ((1 - self.x) ** (b - 1))) / ((math.gamma(a)*math.gamma(b))/math.gamma(a+b))

        # return ((math.gamma(alpha+beta))/(math.gamma(alpha)*math.gamma(beta)))*(y**(alpha-1))*(1-y)**(beta-1)

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

def detType(nAnnot, p_fo, p_KG_u):
    rans = [random.random() for _ in range(nAnnot)]
    type = ["normal" if ran > p_fo + p_KG_u
            else "KG" if ran > p_fo
            else "first_only" for ran in rans]
    return type


def createData(path, car_list, modes, dups, p_fos, p_KG_us):
    nAnnot = 20 # 50

    for size in ['small', 'medium', 'large']:
        if size == 'small':
            nQuestions = 80
        elif size == 'medium':
            nQuestions = 200
        else:
            nQuestions = 400
        for car in car_list:
            for mode in modes:
                if mode == 'uniform':
                    param = [['uniform']]
                    distribution = dist(param, x)
                elif mode == 'gaussian':
                    param = [["gaussian", 5, 1]]
                    distribution = dist(param, x)
                elif mode == 'gaussian50_50':
                    param = [["gaussian",2,1],["gaussian",8,1]]
                    distribution = dist(param, x)
                elif mode == "single0":
                    param = [['single', 0]]
                    distribution = dist(param, x)
                elif mode == "single1":
                    param = [['single', 1]]
                    distribution = dist(param, x)
                elif mode[:6] == 'single':
                    param = [['single', mode[6:]]]
                    distribution = dist(param, x)
                elif mode == "beta3_2":
                    param = [['beta', 3,2]]
                    distribution = dist(param, x)
                elif mode == "beta4_2":
                    param = [['beta', 4, 2]]
                    distribution = dist(param, x)
                elif mode == "beta2_4":
                    param = [['beta', 2, 4]]
                    distribution = dist(param, x)
                elif mode == "beta1_3":
                    param = [['beta', 1, 3]]
                    distribution = dist(param, x)
                elif mode == "beta3_1":
                    param = [['beta', 3, 1]]
                    distribution = dist(param, x)
                elif mode == "beta2_2":
                    param = [['beta', 2, 2]]
                    distribution = dist(param, x)
                else:
                    raise ValueError
                for dup in dups:
                    for p_fo in p_fos:
                        for p_KG_u in p_KG_us:
                            udata = {"ID":range(nAnnot),
                                     "type": detType(nAnnot, p_fo,  p_KG_u),
                                     "T_given": random.choices(x, distribution(), k=nAnnot),
                                     "T_model": np.ones(nAnnot)*0.5}

                            for q in range(nQuestions): # keep track of labels in broad format
                                udata[f"q_{q}"] = np.ones(nAnnot)*np.nan

                            user = pandas.DataFrame(data=udata)

                            annotdict = {"ID":range(nQuestions),
                                         "GT": random.choices(range(car), k=nQuestions),
                                         "model": np.zeros(nQuestions)}
                            for i in range(dup):
                                annotdict[f'id_{i}'] = np.zeros(nQuestions)
                                annotdict[f'annot_{i}'] = np.zeros(nQuestions)

                            annotation = pandas.DataFrame(data=annotdict)


                            with Pool(32) as p:
                                results = p.map(partial(dist_annot, user, annotation, dup, car, mode), range(nQuestions))


                            res = np.array([np.concatenate(np.column_stack(i)) for i in results])

                            annotation.loc[:, np.concatenate([[f'id_{i}',f'annot_{i}'] for i in range(dup)])] = res

                            for i, q in enumerate(results):
                                for u_a_pair in zip(q[0],q[1]):
                                    user.loc[u_a_pair[0], f'q_{i}'] = u_a_pair[1]
                            ulen = user.__len__()
                            user = user.drop(np.where(np.all(np.array([np.isnan(user[f'q_{i}']) for i in range(nQuestions)]), axis=0) == True)[0])

                            if user.__len__() != ulen:
                                print(f"warning, user dropped because there were no simulated annotations. user length now: {user.__len__()}")
                            print(f"saving {size}, {car}, {mode}, {dup}, {p_fo}, {p_KG_u}")

                            os.makedirs(f'{path}/simulation data/{mode}/', exist_ok=True)
                            os.makedirs(f'{path}/simulation data/{mode}/csv', exist_ok=True)
                            os.makedirs(f'{path}/simulation data/{mode}/pickle', exist_ok=True)

                            # save data
                            with open(f'{path}/simulation data/{mode}/pickle/{size}_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_KG_u}_user.pickle', 'wb') as file:
                                pickle.dump(user, file)
                            with open(f'{path}/simulation data/{mode}/pickle/{size}_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_KG_u}_annotations_empty.pickle', 'wb') as file:
                                pickle.dump(annotation, file)
if __name__ == "__main__":

    # car_list = list(range(2,8))
    car_list = [3]

    # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    # modes = ['beta2_4', 'beta2_2', 'beta4_2']
    modes = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
    # dups = [3, 5, 7, 9]
    # dups = [2,5,9]
    dups = [3]

    p_fos = [0.0, 0.1]
    # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]
    p_KG_us = [0.0, 0.1]
    # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]
    createData(os.getcwd(), car_list, modes, dups, p_fos, p_KG_us)