import pickle

import matplotlib.pyplot as plt

import pandas
from matplotlib import cm
import numpy as np
from G_EM import EM
from mcmc import mcmc
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.axes3d import get_test_data


def process_u_error(session_folder):

    mcmc_data.loc[:,'U_error'] = np.zeros(mcmc_data.__len__())
    for car in car_list:
        for mode in modes[:1]:
            for dup in dups:
                for p_fo in p_fos:
                    for p_kg in p_kgs:
                        for p_kg_u in p_kg_us:
                            with open(f'data/{session_folder}/mcmc_annotations_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                mcmc_annotations = pickle.load(file)
                            with open(f'data/{session_folder}/mcmc_user_data_{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{iterations["mcmc"]}.pickle', 'rb') as file:
                                mcmc_user = pickle.load(file)
                            # error for normal users: modelled T - GT
                            u_norm_error = sum(abs(mcmc_user.loc[(mcmc_user['type']=='normal'), 'T_model']-mcmc_user.loc[(mcmc_user['type']=='normal'), 'T_given']))

                            # error for known good users: modelled T - 1
                            u_kg_error = sum((mcmc_user.loc[(mcmc_user['type']=='KG'), 'T_model']-1)*-1)

                            # error for malicious users: modelled T - 1/K
                            u_fo_error = sum(abs(mcmc_user.loc[(mcmc_user['type'] == 'KG'), 'T_model'] - (1/car)))

                            mcmc_data.loc[(mcmc_data['iterations'].values == iterations['mcmc']) &
                                          (mcmc_data['car'].values == car) &
                                          (mcmc_data['mode'].values == mode) &
                                          (mcmc_data['dup'].values == dup) &
                                          (mcmc_data['p_fo'].values == p_fo) &
                                          (mcmc_data['p_kg'].values == p_kg) &
                                          (mcmc_data['p_kg_u'].values == p_kg_u), 'U_error'] = (u_norm_error + u_kg_error + u_fo_error)/mcmc_user.__len__()

class plots():
    def __init__(self):
        self.figcar, self.axscar = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)
        self.figuerror, self.axsuerror = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)

    def plot(self):
        for i, dup in enumerate(dups):
            for j, p_fo in enumerate(p_fos):
                # plot naive
                self.axscar[i,j].plot(car_list,em_data.loc[
                    (em_data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in em_data['car']]) &
                    (em_data['mode'] == mode) &
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['dup'] == dup),
                    'pc_n'
                ], label='Naive')

                # plot EM
                self.axscar[i,j].plot(car_list,em_data.loc[
                    (em_data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in em_data['car']]) &
                    (em_data['mode'] == mode)&
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['dup'] == dup),
                    'pc_m'
                ], label='EM')

                # plot MCMC
                self.axscar[i, j].plot(car_list, mcmc_data.loc[
                    (mcmc_data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in mcmc_data['car']]) &
                    (mcmc_data['mode'] == mode) &
                    (mcmc_data['p_fo'] == p_fo) &
                    (mcmc_data['p_kg'] == p_kg) &
                    (mcmc_data['dup'] == dup),
                    'pc_m'
                ], label='MCMC')

                # plot uerror
                self.axsuerror[i, j].plot(car_list, mcmc_data.loc[
                    (mcmc_data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in mcmc_data['car']]) &
                    (mcmc_data['mode'] == mode) &
                    (mcmc_data['p_fo'] == p_fo) &
                    (mcmc_data['p_kg'] == p_kg) &
                    (mcmc_data['dup'] == dup),
                    'pc_m'
                ], label='U_error')

                # self.axscar[i,j].set_title(f'dup={dup}, p_fo={p_fo}')
                if i == 0:
                    self.axscar[i,j].set_title(f'proportion \n first only = {p_fo}')
                    self.axsuerror[i, j].set_title(f'proportion \n first only = {p_fo}')

                if i == dups.__len__()-1:
                    self.axscar[i,j].set_xlabel('Cardinality')
                    self.axsuerror[i,j].set_xlabel('Cardinality')
                    if j == p_fos.__len__()-1:
                        self.axscar[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
                        self.axsuerror[i, j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
                if j == 0:
                    self.axscar[i, j].set_ylabel(f'duplication factor = {dup}\n\n Proportion correct')
                    self.axsuerror[i, j].set_ylabel(f'duplication factor = {dup}\n\n Proportion correct')
        # plt.subplots_adjust(hspace=0.4)

    def plot_interactive(self):

        self.plot()
        self.figcar.subplots_adjust(left=0.25, bottom=0.25)

        ## known good answer ##
        axpkg = self.figcar.add_axes([0.15, 0.25, 0.0225, 0.63])
        def getKGSlider(val):
            pkg_slider = Slider(
                ax=axpkg,
                label="prop known good",
                valmin=0,
                valmax= p_kgs[-1],
                valinit= val,
                valstep= p_kgs,
                orientation="vertical"
            )
            return pkg_slider
        pkg_slider = getKGSlider(p_kgs[0])

        def updatepkg(val):
            global p_kg
            global pkg_slider
            p_kg = val
            for row in self.axscar:
                for col in row:
                    col.clear()
            axpkg.clear()
            self.plot()
            pkg_slider = getKGSlider(val)
            plt.show()

        ## known good user ##
        axpkgu = self.figcar.add_axes([0.10, 0.25, 0.0225, 0.63])
        def getKGUSlider(val):
            pkgu_slider = Slider(
                ax=axpkgu,
                label="prop known good users",
                valmin=0,
                valmax= p_kg_us[-1],
                valinit= val,
                valstep= p_kg_us,
                orientation="vertical"
            )
            return pkgu_slider
        pkgu_slider = getKGUSlider(p_kg_us[0])

        def updatepkgu(val):
            global p_kg_u
            global pkgu_slider
            p_kg_u = val
            for row in self.axscar:
                for col in row:
                    col.clear()
            axpkgu.clear()
            self.plot()
            pkgu_slider = getKGUSlider(val)
            plt.show()

        pkgu_slider.on_changed(updatepkgu)

        ## dataset mode ##
        axmode = self.figcar.add_axes([0.05, 0.25, 0.0225, 0.63])
        modeindex = range(modes.__len__())
        def getModeSlider(val):
            mode_slider = Slider(
                ax=axmode,
                label=f"Simulation distribution: \n {modes[val]}",
                valmin=0,
                valmax= modeindex[-1],
                valinit= val,
                valstep= modeindex,
                orientation="vertical"
            )
            return mode_slider
        mode_slider = getModeSlider(modeindex[0])

        def updatemode(val):
            global mode
            global mode_slider
            mode = modes[val]
            for row in self.axscar:
                for col in row:
                    col.clear()
            axmode.clear()
            self.plot()
            mode_slider = getModeSlider(val)
            plt.show()

        pkg_slider.on_changed(updatepkg)
        mode_slider.on_changed(updatemode)

        plt.show()

if __name__ == "__main__":
    # model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'
    em_sessions = ["session_2022-12-16_11-34-04"]
    mcmc_sessions = ["session_2022-12-13_11-33-52"]

    iterations = {'em':10,
                  'mcmc': 40}
    # car_list = list(range(2, 8))
    car_list = list(range(2, 6))

    modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]                # duplication factor of the annotators
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]

    datalen = 2*car_list.__len__()*modes.__len__()*dups.__len__()*p_fos.__len__()*p_kgs.__len__()*p_kg_us.__len__()
    cols = ['model', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'EM', 'pc_m', 'pc_n', 'uerror']
    data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)

    for session in em_sessions:
        em_filepath = f'data/{session}/em_data_{"_".join(modes)}.pickle'
        with open(em_filepath, 'rb') as file:
            em_data = pickle.load(file)

        # data.loc[:, "em", 10, ]
        process_u_error(session)

    for session in mcmc_sessions:
        mcmc_filepath = f'data/{session}/mcmc_data_{"_".join(modes)}.pickle'
        with open(mcmc_filepath, 'rb') as file:
            mcmc_data = pickle.load(file)
        process_u_error(session)

    plot = plots()

    # inits
    p_kg = p_kgs[0]
    mode = modes[0]
    # iterations = iterations_list[0]

    plot.plot_interactive()


    # for iterations in iterations_list:
    #     for car in car_list:
    #         for mode in modes:
    #             for p_kg in p_kgs:

                    #
                    # plt = plot_car(figdups, axsdups)
                    # figdups.set_size_inches(15, 10)
                    # plt.show()
                    # plt.savefig(latexpath + f'carplot_{model}_p_kg-{p_kg}_data_{mode}_iters-{iterations}.png', dpi=200)
                    # plt.cla()
                    #
                    #
                    # plt = plot_duplication_factor(figcar, axscar)
                    # plt.subplots_adjust(hspace=0.4)
                    # # fig = plt.gcf()
                    # figcar.set_size_inches(15, 10)
                    # plt.savefig(latexpath + f'dupplot_{model}_p_kg-{p_kg}_data_{mode}_iters-{iterations}.png', dpi=200)
                    # plt.clf()



