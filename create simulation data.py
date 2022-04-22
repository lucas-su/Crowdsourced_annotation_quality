import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
import  random

def gamma_dist(alpha, beta, x):
    return 1/(beta**(alpha)*math.gamma(alpha))*x**(alpha-1)*math.e**(-x/beta)

def beta_dist(alpha, beta, x, xmax):
    x = (1/xmax)*x # normalize to entire space instead of 0,1
    return ((math.gamma(alpha+beta))/(math.gamma(alpha)*math.gamma(beta)))*(x**(alpha-1))*(1-x)**(beta-1)

def gauss_dist(mu, sd, x):
    return (math.e ** (-((x - mu) ** 2) / 2 * (sd) ** (2))) / sd * math.sqrt(2 * math.pi)

def norm(x, xmax, steps):
    int = sum(x * (xmax / steps))
    return x/int

def plot_dists():
    fig, ax = plt.subplots(2,1)

    ax[0].plot(x, gam_beta_mean, linewidth=2.0)
    ax[0].plot(x, gam_gauss_mean, linewidth=2.0)

    ax[1].plot(x, gam_y, linewidth=2.0)
    ax[1].plot(x, gauss_y, linewidth=2.0)
    ax[1].plot(x, beta_y, linewidth=2.0)

    ax[0].set(xlim=(0, xmax), xticks=np.arange(1, xmax),
           ylim=(0, 1), yticks=np.arange(0.1, 1, 0.1))
    ax[1].set(xlim=(0, xmax), xticks=np.arange(1, xmax),
           ylim=(0, 1), yticks=np.arange(0.1, 1, 0.1))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,10,samples.__len__()),samples, 'o')
    plt.show()

if __name__ == "__main__":
    # global model parameters
    Gamma = {'alpha': 1, 'beta': 1}
    Beta = {'alpha': 13, 'beta': 13}
    mu = 5
    sd = 1

    # other globals
    xmax = 10
    steps = 100
    x = np.linspace(0, xmax, steps)

    # datasets
    users = pandas.DataFrame(columns=['id', 'quality'])
    annotations = pandas.DataFrame(columns=['label_1', 'id_1', 'label_2', 'id_2', 'label_3', 'id_3', 'ground_truth'])

    # individual probability distributions
    gam_y = gamma_dist(Gamma['alpha'],Gamma['beta'],x)
    gauss_y = gauss_dist(mu, sd, x)
    beta_y = beta_dist(Beta['alpha'], Beta['beta'], x, xmax)

    # combined probability distributions
    gam_gauss_mean = (gam_y + gauss_y / 4) / 2
    gam_beta_mean = (gam_y + beta_y / 4) / 2

    # normalize to produce p=1 in total
    gam_y = norm(gam_y, xmax, steps)
    gauss_y = norm(gauss_y, xmax, steps)
    beta_y = norm(beta_y, xmax, steps)
    gam_gauss_mean = norm(gam_gauss_mean, xmax, steps)
    gam_beta_mean = norm(gam_beta_mean, xmax, steps)

    # select samples from distribution with k samples
    samples = random.choices(x, gam_beta_mean, k=10 ** 3)

    # plot dists (turn off for debugging, messes with debugger)
    # plot_dists()

    # fill in users and quality
    users.id = np.arange(0,1000,1)
    users.quality = samples
    annotations.ground_truth = np.ones(100000)

    while (annotations[pandas.isna(annotations.id_1)].__len__()>0) and (annotations[pandas.isna(annotations.id_2)].__len__()>0) and (annotations[pandas.isna(annotations.id_3)].__len__()>0):
        # select random user
        user = random.randint(0,users.__len__()-1)
        # select index which user can write
        index = annotations.loc[
                                (annotations.id_1 != user) &
                                (annotations.id_2 != user) &
                                (annotations.id_3 != user) &
                                (
                                    (pandas.isna(annotations.id_1)) |
                                    (pandas.isna(annotations.id_2)) |
                                    (pandas.isna(annotations.id_3))
                                )
                                ].index[0]
        # write in annotation with correct label based on annotator quality
        if (annotations.id_1.iloc[index] != user) & (pandas.isna(annotations.id_1.iloc[index])):
            annotations.loc[index,'id_1'] = user
            annotations.loc[index, 'label_1'] = 1 if users[users.id==user].quality.values[0] > random.random() else 0
        elif (annotations.id_2.iloc[index] != user) & (pandas.isna(annotations.id_2.iloc[index])):
            annotations.loc[index, 'id_2'] = user
            annotations.loc[index, 'label_2'] = 1 if users[users.id==user].quality.values[0] > random.random() else 0
        elif (annotations.id_3.iloc[index] != user) & (pandas.isna(annotations.id_3.iloc[index])):
            annotations.loc[index, 'id_3'] = user
            annotations.loc[index, 'label_3'] = 1 if users[users.id==user].quality.values[0] > random.random() else 0

    print("done, saving")

    # save data
    with open('simulation data/users.pickle', 'wb') as file:
        pickle.dump(users, file)
    with open('simulation data/annotations.pickle', 'wb') as file:
        pickle.dump(annotations, file)