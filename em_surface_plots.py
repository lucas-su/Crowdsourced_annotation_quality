import pickle

import matplotlib.pyplot as plt

import pandas
from matplotlib import cm
import numpy as np
from G_EM import EM

from mpl_toolkits.mplot3d.axes3d import get_test_data

def plot_duplication_factor():
    fig, axs = plt.subplots(car_list.__len__(), p_fos.__len__(), sharex=True, sharey=True)
    for i, car in enumerate(car_list):
        for j, p_fo in enumerate(p_fos):
            axs[i,j].plot(dups,em_data.loc[
                (em_data['iterations'] == iter_select) &
                (em_data['mode'] == 'uniform')&
                (em_data['p_fo'] == p_fo) &
                (em_data['car'] == car),
                'pc_n'
            ], label='Naive')
            axs[i,j].plot(dups,em_data.loc[
                (em_data['iterations'] == iter_select) &
                (em_data['mode'] == mode_select)&
                (em_data['p_fo'] == p_fo) &
                (em_data['car'] == car),
                'pc_m'
            ], label='Modelled')
            axs[i,j].set_title(f'car={car}, p_fo={p_fo}')
            if i == car_list.__len__()-1:
                axs[i,j].set_xlabel('Annotator duplication factor')
                if j == p_fos.__len__()-1:
                    axs[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.08, 1, 1))
            if j == 0:
                axs[i, j].set_ylabel('Prop. correct')
    plt.subplots_adjust(hspace=0.4)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    # plt.savefig(latexpath + 'dup_factor.png', dpi=200)
    plt.show()

def plot_car():
    fig, axs = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)

    for i, dup in enumerate(dups):
        for j, p_fo in enumerate(p_fos):
            axs[i,j].plot(car_list,em_data.loc[
                (em_data['iterations'] == iter_select) &
                pandas.Series([item in car_list for item in em_data['car']]) &
                (em_data['mode'] == mode_select) &
                (em_data['p_fo'] == p_fo) &
                (em_data['dup'] == dup),
                'pc_n'
            ], label='Naive')
            axs[i,j].plot(car_list,em_data.loc[
                (em_data['iterations'] == iter_select) &
                pandas.Series([item in car_list for item in em_data['car']]) &
                (em_data['mode'] == mode_select)&
                (em_data['p_fo'] == p_fo) &
                (em_data['dup'] == dup),
                'pc_m'
            ], label='Modelled')
            axs[i,j].set_title(f'dup={dup}, p_fo={p_fo}')
            if i == dups.__len__()-1:
                axs[i,j].set_xlabel('Cardinality')
                if j == p_fos.__len__()-1:
                    axs[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
            if j == 0:
                axs[i, j].set_ylabel('Proportion correct')
    plt.subplots_adjust(hspace=0.4)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    # plt.savefig(latexpath + 'cardinality_plot.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    latexpath = 'C:\\users\\admin\\pacof\\notes\\Papers\\EM for annotations\\figures\\em surface plots\\'



    iterations_list = [2,3,5]
    car_list = list(range(3,8))
    modes = ['uniform', 'gaussian']
    dups = [3,5,7,9]
    p_fos = [0.0,0.1,0.2,0.3]
    # p_fos = [0.3]
    # iterations_list = [5,10,15,20]
    iter_select = 5
    mode_select = modes[1]
    # car_list = list(range(2,9))
    # modes = ['uniform']
    # dups = [3,5,7,9]
    # p_fos = [0.0,0.1,0.2,0.3]

    with open('data/em_data_uniform_gaussian.pickle', 'rb') as file:
        em_data = pickle.load(file)
    plot_car()
    plot_duplication_factor()



