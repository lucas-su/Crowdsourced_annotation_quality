import os.path
import pickle
from settings import *
import matplotlib.pyplot as plt
import create_stats
import pandas
from matplotlib import cm
import numpy as np
from em import EM
from mcmc import *
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.axes3d import get_test_data





class plots():
    def __init__(self):
        self.figcar, self.axscar = plt.subplots(dup_list.__len__(), p_fo_list.__len__(), sharex=True, sharey=True)
        self.figuerror, self.axsuerror = plt.subplots(dup_list.__len__(), p_fo_list.__len__(), sharex=True, sharey=True)
        self.figsave, self.axssave = plt.subplots()
        self.figsave.set_size_inches(8,4)
        self.figsave.subplots_adjust(top=0.98, bottom=0.12,right=0.83,left=0.08)

    def plot_one(self, mode, dup, p_fo, p_kg, iterations, datasetsize, p_kg_u):
        self.axssave.plot(car_list, data.loc[
            (data['size'] == datasetsize) &
            (data['model'] == 'mcmc') &
            (data['iterations'] == iterations['mcmc']) &
            pandas.Series([item in car_list for item in data['car']]) &
            (data['mode'] == mode) &
            (data['p_fo'] == p_fo) &
            (data['p_kg'] == p_kg) &
            (data['p_kg_u'] == p_kg_u) &
            (data['dup'] == dup),
            'pc_n'
        ], label='Maj. vote')

        # plot EM
        self.axssave.plot(car_list, data.loc[
            (data['size'] == datasetsize) &
            (data['model'] == 'em') &
            (data['iterations'] == iterations['em']) &
            pandas.Series([item in car_list for item in data['car']]) &
            (data['mode'] == mode) &
            (data['p_fo'] == p_fo) &
            (data['p_kg'] == p_kg) &
            (data['p_kg_u'] == p_kg_u) &
            (data['dup'] == dup),
            'pc_m'
        ], label='EM')

        # plot MCMC
        self.axssave.plot(car_list, data.loc[
            (data['size'] == datasetsize) &
            (data['model'] == 'mcmc') &
            (data['iterations'] == iterations['mcmc']) &
            pandas.Series([item in car_list for item in data['car']]) &
            (data['mode'] == mode) &
            (data['p_fo'] == p_fo) &
            (data['p_kg'] == p_kg) &
            (data['p_kg_u'] == p_kg_u) &
            (data['dup'] == dup),
            'pc_m'
        ], label='MCMC')

        # plot krippendorf a pruned pc
        self.axssave.plot(car_list, data.loc[
            (data['size'] == datasetsize) &
            (data['model'] == 'mcmc') &
            (data['iterations'] == iterations['mcmc']) &
            pandas.Series([item in car_list for item in data['car']]) &
            (data['mode'] == mode) &
            (data['p_fo'] == p_fo) &
            (data['p_kg'] == p_kg) &
            (data['p_kg_u'] == p_kg_u) &
            (data['dup'] == dup),
            'pc_aftr_prun_total'  # pc_aftr_prun_total pc_aftr_prun
        ], label='Krip. α')
        self.axssave.set_xlabel('Cardinality')
        self.axssave.set_xticks([3,5,7])
        self.axssave.legend(loc='lower left', bbox_to_anchor=(1, 0, 1, 1))
        self.axssave.set_ylabel(f'Proportion correct')



    def saveplots(self):
        # mode, dup, p_fo, p_kg, iterations, size, p_kg_u
        plotdata = [['beta2_4', 2,0,0,iterations,'small', 0],
                    ['beta2_4', 9,0,0,iterations,'small', 0],
                    ['beta4_2', 2, 0, .2, iterations, 'comb', .2],
                    ['beta4_2', 2, 0, .1, iterations, 'large', .1],
                    ['beta2_2', 2, 0, .2, iterations, 'comb', .2],
                    ['beta2_2', 2, 0.2, .2, iterations, 'comb', .2]]
        for plotdatum in plotdata:
            self.plot_one(*plotdatum)
            plt.savefig(f'C:\\Users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\PC_mode-{plotdatum[0]}_dup-{plotdatum[1]}_p_fo-{plotdatum[2]}_p_kg-{plotdatum[3]}_size-{plotdatum[5]}_p_kg_u{plotdatum[6]}.png', dpi=300)
            self.axssave.clear()


    def plot(self):
        # todo make non-global
        for i, dup in enumerate(dup_list):
            for j, p_fo in enumerate(p_fo_list):
                # plot naive
                self.axscar[i,j].plot(car_list,data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'pc_n'
                ], label='Maj. vote')

                # plot EM
                self.axscar[i,j].plot(car_list,data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'em') &
                    (data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'pc_m'
                ], label='EM')

                # plot em uerror
                self.axsuerror[i, j].plot(car_list, data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'em') &
                    (data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'uerror'
                ], label='EM')


                # plot MCMC
                self.axscar[i, j].plot(car_list, data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'pc_m'
                ], label='MCMC')

                # plot mcmc uerror
                self.axsuerror[i, j].plot(car_list, data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'uerror'
                ], label='MCMC')

                # plot krippendorf a pruned pc
                self.axscar[i, j].plot(car_list, data.loc[
                    (data['size'] == datasetsize) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == T_dist) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'pc_aftr_prun_total' # pc_aftr_prun_total pc_aftr_prun
                ], label='Krip a')

                # self.axscar[i,j].set_title(f'dup={dup}, p_fo={p_fo}')
                if i == 0:
                    self.axscar[i,j].set_title(f'proportion \n first only = {p_fo}')
                    self.axsuerror[i, j].set_title(f'proportion \n first only = {p_fo}')

                if i == dup_list.__len__()-1:
                    self.axscar[i,j].set_xlabel('Cardinality')
                    self.axsuerror[i,j].set_xlabel('Cardinality')
                    if j == p_fo_list.__len__()-1:
                        self.axscar[i,j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
                        self.axsuerror[i, j].legend(loc='lower left', bbox_to_anchor=(1, -0.04, 1, 1))
                if j == 0:
                    self.axscar[i, j].set_ylabel(f'duplication factor = {dup}\n\n Proportion correct')
                    self.axsuerror[i, j].set_ylabel(f'duplication factor = {dup}\n\n Trustworthiness error')
        # plt.subplots_adjust(hspace=0.4)

    def plot_interactive(self):

        self.plot()
        self.figcar.subplots_adjust(left=0.25, bottom=0.25)

        ## known good answer ##
        axpkg = self.figcar.add_axes([0.15, 0.25, 0.0225, 0.63])
        def getKGSlider(val):
            pkg_slider = Slider(
                ax=axpkg,
                label="proportion\nknown good\nitems",
                valmin=0,
                valmax= p_kg_list[-1],
                valinit= val,
                valstep= p_kg_list,
                orientation="vertical"
            )
            return pkg_slider
        pkg_slider = getKGSlider(p_kg_list[0])

        def updatepkg(val):
            global p_kg
            global pkg_slider
            p_kg = val
            for row in self.axscar:
                for col in row:
                    col.clear()
            for row in self.axsuerror:
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
                label="proportion\nknown good\nusers",
                valmin=0,
                valmax= p_kg_u_list[-1],
                valinit= val,
                valstep= p_kg_u_list,
                orientation="vertical"
            )
            return pkgu_slider
        pkgu_slider = getKGUSlider(p_kg_u_list[0])

        def updatepkgu(val):
            global p_kg_u
            global pkgu_slider
            p_kg_u = val
            for row in self.axscar:
                for col in row:
                    col.clear()
            for row in self.axsuerror:
                for col in row:
                    col.clear()
            axpkgu.clear()
            self.plot()
            pkgu_slider = getKGUSlider(val)
            plt.show()

        pkgu_slider.on_changed(updatepkgu)

        ## dataset mode ##
        axmode = self.figcar.add_axes([0.05, 0.25, 0.0225, 0.63])
        modeindex = range(T_dist_list.__len__())
        def getModeSlider(val):
            mode_slider = Slider(
                ax=axmode,
                label=f"Simulation\ndistribution:\n{T_dist_list[val]}",
                valmin=0,
                valmax= modeindex[-1],
                valinit= val,
                valstep= modeindex,
                orientation="vertical"
            )
            return mode_slider
        mode_slider = getModeSlider(modeindex[0])

        def updatemode(val):
            global T_dist
            global mode_slider
            T_dist = T_dist_list[val]
            for row in self.axscar:
                for col in row:
                    col.clear()
            for row in self.axsuerror:
                for col in row:
                    col.clear()
            axmode.clear()
            self.plot()
            mode_slider = getModeSlider(val)
            plt.show()

        pkg_slider.on_changed(updatepkg)
        mode_slider.on_changed(updatemode)

        plt.show()

    def plot_pc_T(self, car, dup, p_fo, p_kg, datasetsize, p_kg_u):
        self.figpc_T, self.axspc_T = plt.subplots()
        mcmcpc_m = data.loc[(data['model'] == 'mcmc') &
                        (data['car'] == car) &
                        (data['dup'] == dup) &
                        (data['p_fo'] == p_fo) &
                        (data['p_kg'] == p_kg) &
                        # (data['size'] == datasetsize) &
                        (data['p_kg_u'] == p_kg_u), 'pc_m']
        mcmc_sd = data.loc[(data['model'] == 'mcmc') &
                            (data['car'] == car) &
                            (data['dup'] == dup) &
                            (data['p_fo'] == p_fo) &
                            (data['p_kg'] == p_kg) &
                            # (data['size'] == datasetsize) &
                            (data['p_kg_u'] == p_kg_u), 'pc_m_SD']
        # empc_m = data.loc[(data['model'] == 'em') &
        #                     (data['car'] == car) &
        #                     (data['dup'] == dup) &
        #                     (data['p_fo'] == p_fo) &
        #                     (data['p_kg'] == p_kg) &
        #                     (data['size'] == size) &
        #                     (data['p_kg_u'] == p_kg_u), 'pc_m']
        # em_sd = data.loc[(data['model'] == 'em') &
        #                    (data['car'] == car) &
        #                    (data['dup'] == dup) &
        #                    (data['p_fo'] == p_fo) &
        #                    (data['p_kg'] == p_kg) &
        #                    (data['size'] == size) &
        #                    (data['p_kg_u'] == p_kg_u), 'pc_m_SD']
        naivepc = data.loc[(data['model'] == 'mcmc') &
                            (data['car'] == car) &
                            (data['dup'] == dup) &
                            (data['p_fo'] == p_fo) &
                            (data['p_kg'] == p_kg) &
                            # (data['size'] == datasetsize) &
                            (data['p_kg_u'] == p_kg_u), 'pc_n']
        naive_sd = data.loc[(data['model'] == 'mcmc') &
                           (data['car'] == car) &
                           (data['dup'] == dup) &
                           (data['p_fo'] == p_fo) &
                           (data['p_kg'] == p_kg) &
                        #    (data['size'] == datasetsize) &
                           (data['p_kg_u'] == p_kg_u), 'pc_n_SD']
        x = np.arange(11)/10

        self.axspc_T.plot(x, mcmcpc_m, label='mcmc')
        self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(mcmcpc_m+mcmc_sd, dtype=float)], [max(sd, 0) for sd in np.array(mcmcpc_m-mcmc_sd, dtype=float)], alpha=0.2)
        # self.axspc_T.plot(x, empc_m, label='em')
        # self.axspc_T.fill_between(x, np.array(empc_m + em_sd, dtype=float), np.array(empc_m - em_sd, dtype=float), alpha=0.2)
        self.axspc_T.plot(x, naivepc, label='maj. vote')
        self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(naivepc + naive_sd, dtype=float)], [max(sd, 0) for sd in np.array(naivepc - naive_sd, dtype=float)], alpha=0.2 )
        self.axspc_T.hlines(1/car, x[0], x[-1], label='1/cardinality', colors='green')
        self.axspc_T.set_xlabel('Proportion T=1 vs. T=0')
        self.axspc_T.set_ylabel('Proportion item labels correct')
        self.axspc_T.set_title(f'Prop. of items correct for car {car}, duplication factor {dup}, known good items prop. {p_kg}, datasetsize {datasetsize}, known good users prop. {p_kg_u}')
        self.axspc_T.legend()
        plt.show()

if __name__ == "__main__":
    # model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'

    iterations = {'em':10,
                  'mcmc': 100}
    for size in datasetsize_list:
        for car in car_list:
            for dup in dup_list:
                for p_fo in p_fo_list:
                    for p_kg in p_kg_list:
                        for p_kg_u in p_kg_u_list:
                            if not os.path.exists(f'{set_session_dir(size, car, dup, p_fo, p_kg, p_kg_u)}/stats.pickle'):
                                create_stats.main(size, car, dup, p_fo, p_kg, p_kg_u)




    # data['size'] = datasetsize

  
    plot = plots()

    # inits
    # p_kg = p_kg_list[0]
    # p_fo = p_fo_list[0]
    # p_kg_u = p_kg_u_list[0]
    # T_dist = T_dist_list[0]
    # dup = dup_list[0]


    # plot.saveplots()
    # plot.plot_interactive()
    for size in datasetsize_list:
        for car in car_list:
            for p_kg in p_kg_list:
                for p_kg_u in p_kg_u_list:
                    for dup in dup_list:
                        for p_fo in p_fo_list:
                            with open(f'{set_session_dir(size, car, dup, p_fo, p_kg, p_kg_u)}/stats.pickle','rb') as file:
                                data = pickle.load(file)
                            data = data.loc[data['session'] == 'avg']
                            plot.plot_pc_T(car, dup, p_fo, p_kg, size, p_kg_u)
