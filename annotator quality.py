import pandas, pickle
import numpy as np
from statistics import  mean, stdev
from multiprocessing import Pool
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


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
                for otherannot in np.arange(1, 4):
                    if annotnum != otherannot:
                        if annotations.loc[currentAnnotationIndex, f'label_{annotnum}'] == \
                                annotations.loc[currentAnnotationIndex, f'label_{otherannot}']:
                            total_agreement += \
                            users.loc[users.id == annotations.loc[currentAnnotationIndex, f'id_{otherannot}']][
                                f'q_weight_{iteration}'].values[0]
                        else:
                            pass
                        total_messages += 1
    return total_agreement, total_messages

def normalize(method='identity'):
    global users
    user_weights_unnormalized = users.loc[
        users.included == 1, f'q_weight_{iteration}']
    if method == 'mean':
        # rebalance based on mean
        users.loc[users.included == 1, f'q_weight_{iteration}'] = user_weights_unnormalized / mean_weight
    elif method == 'softmax':
        # softmax rebalance
        users.loc[users.included == 1, f'q_weight_{iteration}'] = (np.exp(user_weights_unnormalized)*user_weights_unnormalized.__len__() / np.sum(np.exp(user_weights_unnormalized)))
    elif method == 'identity':
        # identity rebalance
        users.loc[users.included == 1, f'q_weight_{iteration}'] = (user_weights_unnormalized * user_weights_unnormalized.__len__()) / np.sum(user_weights_unnormalized)
    else:
        raise ValueError("no method set in normalize")

def remove_users(method='mean'):
    global users
    if method == 'mean':
        # mark users as excluded if their weight is higher than one SD below mean
        users['included'] = [iteration * -1
                                if weight < (mean_weight - (sd_weight * sd_multiplier))
                                else u_inc
                                    for weight, u_inc
                                    in zip(users[f'q_weight_{iteration}'], users.included)]
    else:
        weights = np.reshape(list(users.loc[users.included == 1, f'q_weight_{iteration}']),(-1,1))

        if method == 'elliptic':
            if iteration >0 :
                include = EllipticEnvelope(random_state=0, contamination=0.08).fit(weights).predict(weights)
            else:
                include = np.ones(1000)
        elif method == 'localoutlier':
            clf = LocalOutlierFactor(n_neighbors=20)
            include = clf.fit_predict(weights)
        elif method == 'OCSVM':
            if iteration >0:
                clf = OneClassSVM(kernel='linear', nu=0.03).fit(weights)
                include = clf.predict(weights)
            else:
                include = np.ones(1000)
        else:
            raise ValueError

        for i, user in enumerate(users.loc[users.included == 1].id):
            if include[i] == -1:
                users.loc[user, "included"] = iteration * -1



if __name__ == "__main__":
    iterations = 12                 # max iterations, should stop before this
    sd_multiplier = 1               # for use in mean removal method
    stopping_sd = 0.08              # 0.1 works well
    normalizeMethod = 'softmax'     # options: mean softmax identity
    removeMethod = 'OCSVM'          # options: elliptic localoutlier OCSVM

    with open('simulation data/users.pickle', 'rb') as file:
        users = pickle.load(file)
    with open('simulation data/annotations.pickle', 'rb') as file:
        annotations = pickle.load(file)

    for i in range(iterations+1):
        users[f'q_weight_{i}'] = np.ones(users.__len__()) # all users start at weight 1, weight goes from 0-2
    users['included'] = np.ones(users.__len__())

    for iteration in np.arange(0, iterations):
        mean_weight = mean(users.loc[users.included == 1, f'q_weight_{iteration}'])
        sd_weight = stdev(users.loc[users.included == 1, f'q_weight_{iteration}'])

        print(f"iteration {iteration}")
        print(f'mean: {mean_weight} stdev {sd_weight}')

        normalize(method=normalizeMethod)

        nusers_before_removal = sum([i if i > 0 else 0 for i in users.included])
        remove_users(method=removeMethod)
        nusers_after_removal =  sum([i if i > 0 else 0 for i in users.included])

        print(f"removed users this round: {nusers_before_removal-nusers_after_removal}")
        print(f"users still remaining: {nusers_after_removal}")

        userindices = list(users.loc[(users.included == 1)].id)
        with Pool(16) as p:
            results = p.map(dist_eval, userindices)

        for i, result in enumerate(results):
            total_agreement = result[0]
            total_messages = result[1]
            currentUser = userindices[i]
            if total_messages == 0:
                users.loc[currentUser,f'q_weight_{iteration+1}'] = users.loc[currentUser,f'q_weight_{iteration}']
                print(f'annotator {currentUser} has no co annotators left!')
            else:
                users.loc[currentUser,f'q_weight_{iteration+1}'] = total_agreement/total_messages *2

        if (iteration > 0) & (sd_weight < stopping_sd):
            if iterations-iteration>1:
                users.drop([f'q_weight_{i}' for i in range(iteration+2,iterations+1)], axis=1, inplace=True)
            break
    with open('simulation data/users_with_scores.pickle', 'wb') as file:
        pickle.dump(users, file)
