from sklearn import mixture
import pandas, pickle
import numpy as np
from statistics import  mean, stdev
from multiprocessing import Pool
from dist_eval import dist_eval
from functools import partial


def save_meta_params(model_, sd):
    meta.loc[0, f'mean1_{iteration + 1}'] = model_.means_[0][0]
    meta.loc[0, f'var1_{iteration + 1}'] = sd[0]
    meta.loc[0, f'mean2_{iteration + 1}'] = model_.means_[1][0]
    meta.loc[0, f'var2_{iteration + 1}'] = sd[1]


def compute_weights_bayes(data_):
    model = mixture.GaussianMixture(2)
    model_ = model.fit(data_)
    sd = [np.sqrt(np.trace(model_.covariances_[i]) / 2) for i in range(0, 2)]
    save_meta_params(model_, sd)

    model_high = model_.means_[0][0] < model_.means_[1][0]

    if model_high: # determine which gaussian represents the good annotators
        paB, paG = np.array([model_.predict_proba([result])[0] for result in data_]).T
    else:
        paG, paB = np.array([model_.predict_proba([result])[0] for result in data_]).T

    pG = sum(paG)/paG.__len__()
    pB = sum(paB)/paB.__len__()
    pa = paG * pG + paB * pB
    pGa = (paG*pG)/pa
    return pGa, model_


if __name__ == "__main__":
    iterations = 5

    with open('simulation data/50_50_users_spread.pickle', 'rb') as file:
        users = pickle.load(file)

    # init user weights at 1
    for i in range(iterations+1):
        users[f'q_weight_{i}'] = np.ones(users.__len__()) * 0.5 # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time
    users['included'] = np.ones(users.__len__())

    with open('simulation data/50_50_annotations_spread_empty.pickle', 'rb') as file:
        annotations = pickle.load(file)

    meta = pandas.DataFrame()
    # init with nan because there is no distribution for iteration 0, as all weights are 1
    meta.loc[0, f'mean1_{0}'] = np.nan
    meta.loc[0, f'var1_{0}'] = np.nan
    meta.loc[0, f'mean2_{0}'] = np.nan
    meta.loc[0, f'var2_{0}'] = np.nan
    # set parameter values for plotting
    users.loc[0, f'rel_0'] = np.nan

    for iteration in np.arange(0, iterations):
        mean_weight = mean(users.loc[users.included == 1, f'q_weight_{iteration}'])
        sd_weight = stdev(users.loc[users.included == 1, f'q_weight_{iteration}'])

        print(f"iteration {iteration}")
        print(f'mean: {mean_weight} stdev {sd_weight}')

        userindices = list(users.loc[(users.included == 1)].id)
        with Pool(16) as p:
            results = p.map(partial(dist_eval,annotations, users, iteration), userindices)

        data_ = np.array([result[0]/result[1] for result in results]).reshape(-1,1)
        users.loc[:, f'rel_{iteration + 1}'] = np.array(data_.squeeze()).T
        all_weights, model_ = compute_weights_bayes(data_)

        users.loc[:, f'q_weight_{iteration + 1}'] = np.array(all_weights).T

        weight_diffs = abs(users.loc[:,f"q_weight_{iteration + 1}"] -  users.loc[:,f"q_weight_{iteration}"])
        print(f'Weight diffs above 0.1: {weight_diffs[weight_diffs>0.1].count()}')
        if (iteration > 0) & (weight_diffs[weight_diffs>0.1].count() == 0):
            if iterations-iteration>1:
                users.drop([f'q_weight_{i}' for i in range(iteration+2,iterations+1)], axis=1, inplace=True)
            break

    with open(f'simulation data/users_with_scores.pickle', 'wb') as file:
        pickle.dump(users, file)
    with open(f'simulation data/meta.pickle', 'wb') as file:
        pickle.dump(meta, file)
