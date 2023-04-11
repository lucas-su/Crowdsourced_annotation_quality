import math
import os
import pickle
from multiprocessing import Pool
from functools import partial
import platform
import settings
import matplotlib.pyplot as plt
import numpy as np
import pandas
import  random
from settings import c_data_mal_T
from scipy.stats import beta

from numpy.random import default_rng
rng = default_rng()

# set globals
xmax = 1
steps = 1000
x = np.linspace(0, xmax, steps)


# change t=0 def, larger dataset

def dist_annot(users, annotations, dup, car, T_dist, q_id):
    userlist = np.zeros(dup)

    while not (len(set(userlist)) == len(userlist)):
        userlist = [random.randint(0,users.__len__()-1) for _ in range(dup)]
    answers = [sim_answer(users, annotations, u_id, car, q_id, T_dist) for u_id in userlist]

    return  userlist, answers

def sim_answer(users, annotations, u_id, car, q_id, T_dist):
    if users.loc[u_id].type == "first_only" and T_dist[:6] != "single":
        ans = 0
    elif users.loc[u_id].type == "KG" and T_dist != "single0":
        ans = annotations.loc[q_id,"GT"]
    else:
        # correct answer if trustworthiness is higher than a randomly drawn number, if not a random other answer
        if c_data_mal_T:
            ans = annotations.loc[q_id,"GT"] if users.loc[users.ID==u_id].T_given.values.item() > (random.random()) else \
            random.choice(list(set(np.arange(0,car)) - {annotations.loc[q_id, "GT"]}))
        else:
            ans = annotations.loc[q_id,"GT"] if users.loc[users.ID==u_id].T_given.values.item() > (random.random()) else \
            random.randint(0,car-1) # use randint if 0 trustworthiness means chance level

    return ans

class dist():
    def __init__(self, param, x):
        self.params = param
        self.x = x
        self.dist = None

    def __call__(self, *args, **kwargs):
        return

    def build(self):
        # cum = np.zeros(self.x.__len__())
        for param in self.params:
            func = eval(f'self.{param[0]}') # eval to determine which distribution to take. Options: gamma beta gaussian uniform
            if func.__name__ not in ['gamma', 'beta', 'gaussian', 'uniform', 'single', 'propT_']:
                raise NotImplementedError

        return

    def single(self, prob):
        probs = np.zeros(self.x.shape[0])
        if float(prob) == 1.:
            probs[-1] = 1
        else:
            probs[int(float(prob) * self.x.shape[0])] = 1
        return probs

    def gamma(self, alpha, beta):
        return 1/(beta**(alpha)*math.gamma(alpha))*self.x**(alpha-1)*math.e**(-self.x/beta)

    def beta(self, a,b):
        return beta.pdf(self.x, a,b)
        # return (((self.x) ** (a - 1)) * ((1 - self.x) ** (b - 1))) / ((math.gamma(a)*math.gamma(b))/math.gamma(a+b)) # same thing but my own function doesn't cope with low a and b well, produces overflow warning

    def gaussian(self, mu, sd):
        return (math.e ** (-((self.x - mu) ** 2) / 2 * (sd) ** (2))) / sd * math.sqrt(2 * math.pi)

    def uniform(self):
        return self.x * (max(self.x) / self.x.__len__())
    
    def propT_(self, prop):
        probs = np.zeros(self.x.shape[0])
        prop = float(prop)
        probs[0] = 1-prop
        probs[-1] = prop
        return probs

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

def detType(nAnnot, p_fo):

    rans = [random.random() for _ in range(nAnnot)]
    type = ["normal" if ran > p_fo
            else "first_only" for ran in rans]
    if (p_fo>0) and not ('first_only' in type):
        type = detType(nAnnot, p_fo)
    return type


def createData(path, car, T_dist, dup, p_fo, kg_u, ncpu, size):
    nAnnot = settings.nAnnotator

    if size == 'debug':
        nQuestions = 2*nAnnot
    elif size == 'small':
        nQuestions = nAnnot*3
    elif size == 'medium':
        nQuestions = nAnnot*6
    elif size == 'large':
        nQuestions = nAnnot*20
    elif size == 'xlarge':
        nQuestions = nAnnot * 50
    else:
        raise ValueError

    if T_dist == 'uniform':
        t_given = rng.uniform(0,1,10)
    elif T_dist == 'gaussian':
        t_given = rng.normal(0.5,0.2,10)
    elif T_dist == 'gaussian50_50':
        param = [["gaussian",2,1],["gaussian",8,1]]
        distribution = dist(param, x)
        t_given = random.choices(x, distribution(), k=nAnnot)
    elif T_dist[:6] == 'single':
        t_given = np.linspace(T_dist[6:],T_dist[6:],nAnnot)
    elif T_dist[:5] == "beta2":
        t_given = rng.beta(float(T_dist[5:T_dist.index('_')]),float(T_dist[T_dist.index('_')+1:]), nAnnot)
    elif T_dist[:4] == "beta":
        t_given = rng.beta(float(T_dist[4:T_dist.index('_')]),float(T_dist[T_dist.index('_')+1:]), nAnnot)
    elif T_dist[:6] == 'propT_':
        probs = np.zeros(x.shape[0])
        prop = float(T_dist[6:])
        probs[0] = 1-prop
        probs[-1] = prop
        t_given = random.choices(x, probs, k=nAnnot)
    else:
        raise ValueError
    udata = {"ID":range(nAnnot),
                "type": detType(nAnnot, p_fo),
                "T_given": t_given,
                "T_model": np.ones(nAnnot)*0.5}

    for q in range(nQuestions): # keep track of labels in broad format
        udata[f"q_{q}"] = np.ones(nAnnot)*np.nan

    user = pandas.DataFrame(data=udata)

    kg_u_cnt = 0
    for i, u in (user['type']=='normal').iteritems():
        if kg_u == kg_u_cnt:
            break
        if u:
            user.loc[i, 'type'] = 'KG'
            user.loc[i, 'T_given'] = 1
            kg_u_cnt += 1


    annotdict = {"ID":range(nQuestions),
                    "GT": random.choices(range(car), k=nQuestions),
                    "model": np.zeros(nQuestions),
                    "alpha": [[] for _ in range(nQuestions)],
                    "car": np.ones(nQuestions)*car}
    for i in range(dup):
        annotdict[f'id_{i}'] = np.zeros(nQuestions)
        annotdict[f'annot_{i}'] = np.zeros(nQuestions)

    annotation = pandas.DataFrame(data=annotdict)


    with Pool(ncpu) as p:
        results = p.map(partial(dist_annot, user, annotation, dup, car, T_dist), range(nQuestions))


    res = np.array([np.concatenate(np.column_stack(i)) for i in results])

    annotation.loc[:, np.concatenate([[f'id_{i}',f'annot_{i}'] for i in range(dup)])] = res

    for i, q in enumerate(results):
        for u_a_pair in zip(q[0],q[1]):
            user.loc[u_a_pair[0], f'q_{i}'] = u_a_pair[1]
    ulen = user.__len__()
    user = user.drop(np.where(np.all(np.array([np.isnan(user[f'q_{i}']) for i in range(nQuestions)]), axis=0) == True)[0])

    if user.__len__() != ulen:
        print(f"warning, user dropped because there were no simulated annotations. user length now: {user.__len__()}")
    # print(f"saved annotations | Datasetsize {size}, cardinality {car}, distribution {T_dist}, annotations per item {dup}, prop. malicious {p_fo}, prop. known good users {kg_u}")

    os.makedirs(f'{path}/simulation data/{T_dist}/', exist_ok=True)
    os.makedirs(f'{path}/simulation data/{T_dist}/csv', exist_ok=True)
    os.makedirs(f'{path}/simulation data/{T_dist}/pickle', exist_ok=True)

    # save data
    with open(f'{path}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_user.pickle', 'wb') as file:
        pickle.dump(user, file)
    with open(f'{path}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_annotations_empty.pickle', 'wb') as file:
        pickle.dump(annotation, file)
# if __name__ == "__main__":
#     if platform.system() == 'Windows':
#         ncpu = 8
#     else:
#         ncpu = 32
#     # car_list = list(range(2,8))
#     car_list = [3]
#
#     # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
#     # modes = ['beta2_4', 'beta2_2', 'beta4_2']
#     T_dist_list = [f'single{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]
#     # dups = [3, 5, 7, 9]
#     # dups = [2,5,9]
#     dup_list = [3]
#
#     p_fo_list = [0.0, 0.1]
#     # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]
#     p_KG_u_list = [0.0, 0.1]
#     # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]
#     createData(os.getcwd(), car_list, T_dist_list, dup_list, p_fo_list, p_KG_u_list, ncpu)