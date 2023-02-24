import pybrms


import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
import pprint




def run_model(iterations, car, nQuestions):
    model_df.loc[(model_df['iterations'].values == iterations) &
                 (model_df['car'].values == car) &
                 (model_df['mode'].values == T_dist) &
                 (model_df['dup'].values == dup) &
                 (model_df['p_fo'].values == p_fo), 'model'] = pybrms.fit(formula="count ~ zAge + zBase * Trt + (1 | patient)",
        data=epilepsy,
        family="poisson"
    )




    for q in range(nQuestions):
        k_w = np.zeros(car)
        for k in range(car):

            for d in range(dup):
                if annotations.loc[q, f'annot_{d}'] == k:
                    k_w = [k_w[i] + ((1 - user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                           k_w[i] + user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
                    # k_w[k] += user.loc[annotations.loc[q, f'id_{d}'], 'T_model']
                else:
                    # k_w = [k_w[i]+((1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'])/(car-1)) if i!= k else k_w[i] for i in range(car)]
                    k_w = [
                        k_w[i] + ((user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
                        k_w[i] + 1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
        annotations.loc[q, 'model'] = k_w.index(max(k_w))
    annotations.insert(annotations.columns.get_loc("model") + 1, "naive", np.zeros(nQuestions))
    for q in range(nQuestions):
        k_w = []
        for k in range(car):
            d_w = 0
            for d in range(dup):
                if annotations.loc[q, f'annot_{d}'] == k:
                    d_w += 1
            k_w.append(d_w)
        annotations.loc[q, 'naive'] = k_w.index(max(k_w))

    diff_m = annotations.loc[:, 'GT'] - annotations.loc[:, 'model']
    diff_n = annotations.loc[:, 'GT'] - annotations.loc[:, 'naive']
    diff_m_cnt = (diff_m != 0).sum()
    diff_n_cnt = (diff_n != 0).sum()
    model_df.loc[(model_df['iterations'].values == iterations) &
                 (model_df['car'].values == car) &
                 (model_df['mode'].values == T_dist) &
                 (model_df['dup'].values == dup) &
                 (model_df['p_fo'].values == p_fo), 'pc_m'] = 100 * (1 - (diff_m_cnt / nQuestions))
    model_df.loc[(model_df['iterations'].values == iterations) &
                 (model_df['car'].values == car) &
                 (model_df['mode'].values == T_dist) &
                 (model_df['dup'].values == dup) &
                 (model_df['p_fo'].values == p_fo), 'pc_n'] = 100 * (1 - (diff_n_cnt / nQuestions))
    summary = {"Mode": T_dist,
               "Cardinality": car,
               "Iterations": iterations,
               "Duplication factor": dup,
               "Proportion 'first only'": p_fo,
               "Percentage correct modelled": 100 * (1 - (diff_m_cnt / nQuestions)),
               "Percentage correct naive": 100 * (1 - (diff_n_cnt / nQuestions))}
    [print(f'{key:<30} {summary[key]}') for key in summary.keys()]


if __name__ == "__main__":

    iterations_list = [2,3,5]
    car_list = list(range(3,8))
    # modes = ['uniform', 'gaussian', 'gaussian50_50', 'single0', 'single1', 'beta1_3', 'beta3_1']
    modes = ['single0']
    dups = [3,5,7,9]
    p_fos = [0.0,0.1,0.2,0.3]


    # iterations = 2     # iterations of EM algo
    # car = 5
    # mode = "uniform"    # data modes, options: real, single0 (perfectly bad trustworthiness), single1 (perfect trustworthiness), uniform, gaussian (all except real are simulated)
    # dup = 3             # duplication factor, determines which premade simulation dataset to use
    # p_fo = 0.0          # proportion 'first only' annotators, who are lazy and only ever click the first option
    ###############################


    model_df = pandas.DataFrame(columns=['iterations', 'car', 'mode', 'dup', 'p_fo', 'model', 'pc_m', 'pc_n'])
    for iterations in iterations_list:
        for car in car_list:
            for T_dist in modes:
                for dup in dups:
                    for p_fo in p_fos:
                        # open dataset for selected parameters
                        with open(f'simulation data/{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_user.pickle',
                                  'rb') as file:
                            user = pickle.load(file)
                        with open(
                                f'simulation data/{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_annotations_empty.pickle',
                                'rb') as file:
                            annotations = pickle.load(file)
                        # car = annotations.loc[:,np.concatenate([[f'annot_{i}'] for i in range(dup)])].values.max()+1
                        # init user weights at 1
                        for i in range(iterations + 1):
                            user[f't_weight_{i}'] = np.ones(
                                user.__len__()) * 0.5  # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time
                        user['included'] = np.ones(user.__len__())

                        # nAnnot = user.__len__()
                        nQuestions = annotations.__len__()
                        model_df.loc[model_df.__len__(), :] = [iterations, car, T_dist, dup, p_fo, None, 0, 0]
                        run_model(iterations, car, nQuestions)
                        with open(f'data/user_data_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}.pickle', 'wb') as file:
                            pickle.dump(user, file)
    with open(f'data/em_data_{"_".join(modes)}.pickle', 'wb') as file:
        pickle.dump(model_df, file)
