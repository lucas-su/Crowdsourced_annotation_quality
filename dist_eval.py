import numpy as np

def dist_eval(annotations, users, iteration, currentUser):
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