import pandas, csv, pickle
import numpy as np
from statistics import  mean, stdev


if __name__ == "__main__":
    # globals
    iterations = 3

    with open('simulation data/users.pickle', 'rb') as file:
        users = pickle.load(file)
    with open('simulation data/annotations.pickle', 'rb') as file:
        annotations = pickle.load(file)

    for i in range(iterations+1):
        users[f'q_weight_{i}'] = np.ones(users.__len__()) * 5 # all users start at weight 5
    users['included'] = np.ones(users.__len__())

    for iteration in np.arange(1, iterations+1):
        # determine mean weight of last two iterations
        if iteration > 1:
            users[f'q_weight_{iteration}'] = np.mean(users[f'q_weight_{iteration-1}'], users[f'q_weight_{iteration-2}'])

        # determine means for this iteration
        mean_weight = mean(users[f'q_weight_{iteration}'])
        sd_weight = stdev(users[f'q_weight_{iteration}'])

        print(f"iteration {iteration}")
        print(f'mean: {mean_weight} stdev{sd_weight}')

        # mark users as excluded if their weight is higher than one SD below mean
        users['included'] = [1 if weight > (mean_weight-sd_weight) else 0 for weight in users[f'q_weight_{iteration}']]

        userindices = users.loc[:, users.included == 1]
        for currentUser in userindices:
            total_agreement = 0
            total_messages = 0

            for annotnum in np.arange(1,4):
                allAnnotationsIndices = annotations.loc[
                    (annotations[f'id_{annotnum}'] == currentUser)
                ]
                # if user has been annotator {annotnum} for any label
                if allAnnotationsIndices.__len__() > 0:
                    for currentAnnotationIndex in allAnnotationsIndices:
                        for otherannot in np.arange(1,4):
                            if annotnum != otherannot:
                                if annotations.loc[
                                    (annotations[f'label_{annotnum}'] == currentAnnotationIndex)
                                ] == annotations.loc[
                                    (annotations[f'label_{otherannot}'] == currentAnnotationIndex)
                                ]:
                                    total_agreement += 1
                                else:
                                    pass
                                total_messages += 1


