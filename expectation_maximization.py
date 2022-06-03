from sklearn import mixture
import pandas, pickle
import numpy as np
from statistics import  mean, stdev
from multiprocessing import Pool
# from sklearn.covariance import EllipticEnvelope
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as special

def dist_eval(currentUser):
    """
    Heaviest function is separated to facilitate parallel processing
    """
    total_agreement = 0
    total_messages = 0

    for annotnum in np.arange(1, 4):
        allAnnotationsIndices = annotations.loc[
            (annotations[f'id_{annotnum}'] == currentUser)
        ].index
        # if user has been annotator {annotnum} for any label
        if allAnnotationsIndices.__len__() > 0:
            for currentAnnotationIndex in allAnnotationsIndices:
                # for everybody who annotated current annotation index
                for otherannot in np.arange(1, 4):
                    # if annotator is not current user (don't compare agreement with self)
                    if annotnum != otherannot:
                        # if the annotated label is the same
                        if annotations.loc[currentAnnotationIndex,
                                           f'label_{annotnum}'] == annotations.loc[currentAnnotationIndex, f'label_{otherannot}']:
                            # add weight of other annotator to current annotator agreement
                            total_agreement += \
                            users.loc[users.id == annotations.loc[currentAnnotationIndex, f'id_{otherannot}']][
                                f'q_weight_{iteration}'].values[0]
                        else:
                            pass
                        total_messages += 1
    return total_agreement, total_messages

def plot_dists(means, sd):
    fig, ax = plt.subplots()
    x = np.linspace(min(data_),max(data_),1000)
    ax.plot(data_, np.zeros_like(data_), 'o')
    ax.plot(x, stats.norm.pdf(x, means[0][0], sd[0]))
    ax.plot(x, stats.norm.pdf(x, means[1][0], sd[1]))
    print(f"mean: {means}")
    print(f"sd: {sd}")
    plt.show()

def save_meta_params(model_, sd):
    meta.loc[0, f'mean1_{iteration + 1}'] = model_.means_[0][0]
    meta.loc[0, f'var1_{iteration + 1}'] = sd[0]
    meta.loc[0, f'mean2_{iteration + 1}'] = model_.means_[1][0]
    meta.loc[0, f'var2_{iteration + 1}'] = sd[1]

def compute_weights_1(data_):
    model = mixture.GaussianMixture(2, means_init=[[0.1],[0.8]])
    model_ = model.fit(data_)
    sd = [np.sqrt(np.trace(model_.covariances_[i]) / 2) for i in range(0, 2)]
    save_meta_params(model_, sd)
    # plot_dists(model_.means_, sd)

    all_probs = [model_.predict_proba([result]) for result in data_]

    model_high = model_.means_[0][0] < model_.means_[1][0]
    all_weights = [1+(0.5*prob[0][int(model_high)])-(0.5*prob[0][1-int(model_high)]) for prob in all_probs]
    all_weights = special.softmax(all_weights) * 1000
    return all_weights, model_

def compute_weights_2(data_):
    data_norm = special.softmax(data_) * 1000
    model = mixture.GaussianMixture(2, means_init=[[0.5], [1.5]])

    model_ = model.fit(data_norm)
    sd = [np.sqrt(np.trace(model_.covariances_[i]) / 2) for i in range(0, 2)]
    save_meta_params(model_, sd)
    # plot_dists(model_.means_, sd)

    model_high = model_.means_[0][0] < model_.means_[1][0]
    all_probs = [model_.predict_proba([result]) for result in data_norm]
    all_weights = [1+(0.5*prob[0][int(model_high)])-(0.5*prob[0][1-(model_high)]) for prob in all_probs]
    return all_weights, model_

if __name__ == "__main__":

    iterations = 5
    with open('simulation data/50_50_users_spread.pickle', 'rb') as file:
        users = pickle.load(file)

    # init user weights at 1
    for i in range(iterations+1):
        users[f'q_weight_{i}'] = np.ones(users.__len__()) # all users start at weight 1
    users['included'] = np.ones(users.__len__())

    with open('simulation data/50_50_annotations_spread_empty.pickle', 'rb') as file:
        annotations = pickle.load(file)

    meta = pandas.DataFrame()
    # init with nan because there is no distribution for iteration 0, as all weights are 1
    meta.loc[0, f'mean1_{0}'] = np.nan
    meta.loc[0, f'var1_{0}'] = np.nan
    meta.loc[0, f'mean2_{0}'] = np.nan
    meta.loc[0, f'var2_{0}'] = np.nan

    for iteration in np.arange(0, iterations):
        mean_weight = mean(users.loc[users.included == 1, f'q_weight_{iteration}'])
        sd_weight = stdev(users.loc[users.included == 1, f'q_weight_{iteration}'])

        print(f"iteration {iteration}")
        print(f'mean: {mean_weight} stdev {sd_weight}')

        userindices = list(users.loc[(users.included == 1)].id)
        with Pool(16) as p:
            results = p.map(dist_eval, userindices)

        data_ = np.array([result[0]/result[1] for result in results]).reshape(-1,1)

        all_weights, model_ = compute_weights_1(data_)

        # all_weights, model_ = compute_weights_2(data_)

        for i, weight in enumerate(all_weights):
            users.loc[i, f'q_weight_{iteration + 1}'] = weight

        weight_diffs = abs(users.loc[:,f"q_weight_{iteration + 1}"] -  users.loc[:,f"q_weight_{iteration}"])
        print(f'Weight diffs above 0.1: {weight_diffs[weight_diffs>0.1].count()}')

    with open('simulation data/users_with_scores_1.pickle', 'wb') as file:
        pickle.dump(users, file)
    with open('simulation data/meta.pickle_1', 'wb') as file:
        pickle.dump(meta, file)
