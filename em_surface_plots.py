import pickle

import matplotlib.pyplot as plt

import pandas
from matplotlib import cm
import numpy as np
from G_EM import EM
from mcmc import mcmc
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.axes3d import get_test_data


class plots():
    def __init__(self, model):
        self.figcar, self.axscar = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)
        self.figdups, self.axsdups = plt.subplots(car_list.__len__(), p_fos.__len__(), sharex=True, sharey=True)

    def plot_duplication_factor(self):
        for i, car in enumerate(car_list):
            for j, p_fo in enumerate(p_fos):
                self.axsdups[i,j].plot(dups,em_data.loc[
                    (em_data['iterations'] == iterations) &
                    (em_data['mode'] == mode) &
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['car'] == car),
                    'pc_n'
                ], label='Naive')
                self.axsdups[i,j].plot(dups,em_data.loc[
                    (em_data['iterations'] == iterations) &
                    (em_data['mode'] == mode) &
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['car'] == car),
                    'pc_m'
                ], label='Modelled')
                self.axsdups[i,j].set_title(f'car={car}, p_fo={p_fo}')
                if i == car_list.__len__()-1:
                    self.axsdups[i,j].set_xlabel('Annotator duplication factor')
                    if j == p_fos.__len__()-1:
                        self.axsdups[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.08, 1, 1))
                if j == 0:
                    self.axsdups[i, j].set_ylabel('Prop. correct')

    def plot_car(self):
        print(p_kg)
        for i, dup in enumerate(dups):
            for j, p_fo in enumerate(p_fos):
                self.axscar[i,j].plot(car_list,em_data.loc[
                    (em_data['iterations'] == iterations) &
                    pandas.Series([item in car_list for item in em_data['car']]) &
                    (em_data['mode'] == mode) &
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['dup'] == dup),
                    'pc_n'
                ], label='Naive')
                self.axscar[i,j].plot(car_list,em_data.loc[
                    (em_data['iterations'] == iterations) &
                    pandas.Series([item in car_list for item in em_data['car']]) &
                    (em_data['mode'] == mode)&
                    (em_data['p_fo'] == p_fo) &
                    (em_data['p_kg'] == p_kg) &
                    (em_data['dup'] == dup),
                    'pc_m'
                ], label='Modelled')
                self.axscar[i,j].set_title(f'dup={dup}, p_fo={p_fo}')
                if i == dups.__len__()-1:
                    self.axscar[i,j].set_xlabel('Cardinality')
                    if j == p_fos.__len__()-1:
                        self.axscar[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
                if j == 0:
                    self.axscar[i, j].set_ylabel('Proportion correct')
        # plt.subplots_adjust(hspace=0.4)

    def plot_interactive(self):


        self.plot_car()
        # self.plot_duplication_factor()
        self.figcar.subplots_adjust(left=0.25, bottom=0.25)

        ## iters ##
        axiter = self.figcar.add_axes([0.15, 0.25, 0.0225, 0.63])
        def getIterSlider(val):
            iter_slider = Slider(
                ax=axiter,
                label="iters",
                valmin=0,
                valmax=iterations_list[-1],
                valinit=val,
                valstep=iterations_list,
                orientation="vertical"
            )
            return iter_slider
        iter_slider = getIterSlider(iterations_list[0])

        def updateiter(val):
            global iterations
            global iter_slider
            iterations = val
            # self.figcar.clear()
            # plt.cla()
            for row in self.axscar:
                for col in row:
                    col.clear()
            axiter.clear()
            # for row in self.axsdups:
            #     for col in row:
            #         col.clear()
            self.plot_car()
            # self.plot_duplication_factor()
            iter_slider = getIterSlider(val)
            plt.show()

        ## known good ##
        axpkg = self.figcar.add_axes([0.1, 0.25, 0.0225, 0.63])
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
            # self.figcar.clear()
            # plt.cla()
            for row in self.axscar:
                for col in row:
                    col.clear()
            # for row in self.axsdups:
            #     for col in row:
            #         col.clear()
            axpkg.clear()
            self.plot_car()
            # self.plot_duplication_factor()
            pkg_slider = getKGSlider(val)
            plt.show()


        ## dataset mode ##
        axmode = self.figcar.add_axes([0.05, 0.25, 0.0225, 0.63])
        modeindex = range(modes.__len__())
        def getModeSlider(val):
            mode_slider = Slider(
                ax=axmode,
                label=f"Sim distribution: {modes[val]}",
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
            # self.figcar.clear()
            # plt.cla()
            for row in self.axscar:
                for col in row:
                    col.clear()
            # for row in self.axsdups:
            #     for col in row:
            #         col.clear()
            # self.plot_duplication_factor()
            axmode.clear()
            self.plot_car()
            mode_slider = getModeSlider(val)
            plt.show()

        ## cardinality (for dup plot) ##
        # axcar = self.figcar.add_axes([0.2, 0.25, 0.0225, 0.63])
        #
        # def getCarSlider(val):
        #     car_slider = Slider(
        #         ax=axcar,
        #         label=f"Car {val}",
        #         valmin=car_list[0],
        #         valmax=car_list[-1],
        #         valinit=val,
        #         valstep=car_list,
        #         orientation="vertical"
        #     )
        #     return car_slider
        #
        # car_slider = getCarSlider(modeindex[0])
        #
        # def updatecar(val):
        #     global car
        #     global car_slider
        #     car = val
        #     # self.figcar.clear()
        #     # plt.cla()
        #     for row in self.axscar:
        #         for col in row:
        #             col.clear()
        #     for row in self.axsdups:
        #         for col in row:
        #             col.clear()
        #     self.plot_duplication_factor()
        #     axcar.clear()
        #     self.plot_car()
        #     car_slider = getCarSlider(val)
        #     plt.show()

        # register the update functions
        # car_slider.on_changed(updatecar)
        pkg_slider.on_changed(updatepkg)
        mode_slider.on_changed(updatemode)
        iter_slider.on_changed(updateiter)

        plt.show()

if __name__ == "__main__":
    model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\{model} surface plots\\'

    # car_list = list(range(2, 8))
    car_list = list(range(2, 5))

    iterations_list = [5, 20]

    modes = ['uniform', 'gaussian', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    dups = [3,5,7,9]
    p_fos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    plot = plots(model)
    filepath = f'data/modelled/{model}_data_{"_".join(modes)}.pickle'
    # filepath = 'data/mcmc_data_uniform_gaussian_gaussian50_50_single0_single1_beta1_3_beta3_1.pickle'

    with open(filepath, 'rb') as file:
        em_data = pickle.load(file)


    # inits
    p_kg = p_kgs[0]
    mode = modes[0]
    iterations = iterations_list[0]

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



