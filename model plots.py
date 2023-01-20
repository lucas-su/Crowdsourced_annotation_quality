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
    def __init__(self):
        self.figcar, self.axscar = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)
        self.figuerror, self.axsuerror = plt.subplots(dups.__len__(), p_fos.__len__(), sharex=True, sharey=True)
        self.figsave, self.axssave = plt.subplots()
        self.figsave.set_size_inches(8,4)
        self.figsave.subplots_adjust(top=0.98, bottom=0.12,right=0.83,left=0.08)

    def plot_one(self, mode, dup, p_fo, p_kg, iterations, size, p_kg_u):
        self.axssave.plot(car_list, data.loc[
            (data['size'] == size) &
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
            (data['size'] == size) &
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
            (data['size'] == size) &
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
            (data['size'] == size) &
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
        for i, dup in enumerate(dups):
            for j, p_fo in enumerate(p_fos):
                # plot naive
                self.axscar[i,j].plot(car_list,data.loc[
                    (data['size'] == size) &
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
                self.axscar[i,j].plot(car_list,data.loc[
                    (data['size'] == size) &
                    (data['model'] == 'em') &
                    (data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == mode)&
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'pc_m'
                ], label='EM')

                # plot em uerror
                self.axsuerror[i, j].plot(car_list, data.loc[
                    (data['size'] == size) &
                    (data['model'] == 'em') &
                    (data['iterations'] == iterations['em']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == mode) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'uerror'
                ], label='EM')


                # plot MCMC
                self.axscar[i, j].plot(car_list, data.loc[
                    (data['size'] == size) &
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

                # plot mcmc uerror
                self.axsuerror[i, j].plot(car_list, data.loc[
                    (data['size'] == size) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == mode) &
                    (data['p_fo'] == p_fo) &
                    (data['p_kg'] == p_kg) &
                    (data['p_kg_u'] == p_kg_u) &
                    (data['dup'] == dup),
                    'uerror'
                ], label='MCMC')

                # plot krippendorf a pruned pc
                self.axscar[i, j].plot(car_list, data.loc[
                    (data['size'] == size) &
                    (data['model'] == 'mcmc') &
                    (data['iterations'] == iterations['mcmc']) &
                    pandas.Series([item in car_list for item in data['car']]) &
                    (data['mode'] == mode) &
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

                if i == dups.__len__()-1:
                    self.axscar[i,j].set_xlabel('Cardinality')
                    self.axsuerror[i,j].set_xlabel('Cardinality')
                    if j == p_fos.__len__()-1:
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
        modeindex = range(modes.__len__())
        def getModeSlider(val):
            mode_slider = Slider(
                ax=axmode,
                label=f"Simulation\ndistribution:\n{modes[val]}",
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

# def gentables(data):
#     beta2_4tab = [["",sum(np.where(data.loc[(data['model']=='mcmc') & (data['size']=='mcmc'),'pc_m']>=data.loc[(data['model']=='mcmc'),'pc_n'], 1,0))/sum((data['model']=='mcmc'))]]

if __name__ == "__main__":
    # model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'
    # em_sessions = ["session_2022-12-16_11-34-04"]
    # mcmc_sessions = ["session_2022-12-13_11-33-52"]
    #
    # sessionlen = {'em': em_sessions.__len__(),
    #               'mcmc': mcmc_sessions.__len__()}
    #
    iterations = {'em':10,
                  'mcmc': 40}


    # car_list = list(range(2, 8))
    # modes = ['uniform', 'single0', 'single1', 'beta2_2', 'beta3_2', 'beta4_2']
    # dups = [3,5,7,9]                # duplication factor of the annotators
    # p_fos = [0.0, 0.05, 0.1, 0.15, 0.2]       # proportion 'first only' annotators who only ever select the first option
    # p_kgs = [0.0, 0.05, 0.1, 0.15, 0.2]
    # p_kg_us = [0.0, 0.05, 0.1, 0.15, 0.2]



    car_list = [3, 5, 7]
    modes = ['beta2_4', 'beta2_2', 'beta4_2']
    dups = [2, 5, 9]
    p_fos = [0.0, 0.1, 0.2]
    p_kgs = [0.0, 0.1, 0.2]
    p_kg_us = [0.0, 0.1, 0.2]




    # datalen = 2*car_list.__len__()*modes.__len__()*dups.__len__()*p_fos.__len__()*p_kgs.__len__()*p_kg_us.__len__()
    # cols = ['model', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'EM', 'pc_m', 'pc_n', 'uerror']
    # data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
    # data.loc[:datalen/2,'model'] = "em"
    # data.loc[:datalen / 2, 'iterations'] = 10
    # data.loc[datalen / 2:, 'model'] = "mcmc"
    # data.loc[datalen / 2:, 'iterations'] = 40
    #
    # # init correct values in combined dataframe
    # with open(f'data/{em_sessions[0]}/em_data_{"_".join(modes)}.pickle', 'rb') as file:
    #     tmp_data = pickle.load(file)
    # data.loc[(data['model']=='em'),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = tmp_data.loc[:,['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']]
    #
    # with open(f'data/{mcmc_sessions[0]}/mcmc_data_{"_".join(modes)}.pickle', 'rb') as file:
    #     tmp_data = pickle.load(file)
    # data.loc[(data['model']=='mcmc'),['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']] = tmp_data.loc[:,['car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u']]
    #
    # for session in em_sessions:
    #     em_filepath = f'data/{session}/em_data_{"_".join(modes)}.pickle'
    #     with open(em_filepath, 'rb') as file:
    #         em_data = pickle.load(file)
    #     data.loc[(data['model']=='em'),['pc_m', 'pc_n']] += em_data.loc[:,['pc_m', 'pc_n']]/em_sessions.__len__()
    #     process_u_error(session, 'em')
    #
    # for session in mcmc_sessions:
    #     mcmc_filepath = f'data/{session}/mcmc_data_{"_".join(modes)}.pickle'
    #     with open(mcmc_filepath, 'rb') as file:
    #         mcmc_data = pickle.load(file)
    #     data.loc[(data['model'] == 'mcmc'), ['pc_m', 'pc_n']] += mcmc_data.loc[:, ['pc_m', 'pc_n']]/em_sessions.__len__()
    #     process_u_error(session, 'mcmc')

    # nQuestions = {'small': 80,
    #               'medium': 200,
    #               'large': 400}

    sizes = ['small', 'medium', 'large']
    # datalen = 2*car_list.__len__()*modes.__len__()*dups.__len__()*p_fos.__len__()*p_kgs.__len__()*p_kg_us.__len__()
    # cols = ['model', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'EM', 'pc_m', 'pc_n', 'uerror',
    #         'alpha_bfr_prun', 'n_annot_aftr_prun', 'n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun',
    #         'pc_aftr_prun_total']
    # data = pandas.DataFrame(np.zeros((datalen, cols.__len__())), columns=cols)
    # data.loc[:datalen / 2, 'model'] = "em"
    # data.loc[:datalen / 2, 'iterations'] = 10
    # data.loc[datalen / 2:, 'model'] = "mcmc"
    # data.loc[datalen / 2:, 'iterations'] = 40

    with open(f'exports/data_{sizes[0]}.pickle', 'rb') as file:
        data = pickle.load(file)
    data['size'] = 'small'
    for size in sizes[1:]:
        with open(f'exports/data_{size}.pickle', 'rb') as file:
            tmp_data = pickle.load(file)
            data = pandas.concat((data,tmp_data), ignore_index=True)
            data.loc[data["size"].isnull(), 'size'] = size
    data = pandas.concat((data, tmp_data), ignore_index=True)
    data.loc[data["size"].isnull(), 'size'] = 'comb'
    data.loc[data["size"] == 'comb', ['pc_m', 'pc_n', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun', 'n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total']] = (np.array(data.loc[(data['size']=='small'), ['pc_m', 'pc_n', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun', 'n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total']]) +
                np.array(data.loc[(data['size']=='medium'), ['pc_m', 'pc_n', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun', 'n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total']]) +
                np.array(data.loc[(data['size']=='large'), ['pc_m', 'pc_n', 'uerror', 'alpha_bfr_prun', 'n_annot_aftr_prun', 'n_answ_aftr_prun', 'pc_aftr_prun', 'alpha_aftr_prun', 'pc_aftr_prun_total']]))/3
    plot = plots()

    # inits
    p_kg = p_kgs[0]
    p_kg_u = p_kg_us[0]
    mode = modes[0]
    # iterations = iterations_list[0]

    stats = {"mcmc > naive": sum(np.where(data.loc[(data['model']=='mcmc'),'pc_m']>=data.loc[(data['model']=='mcmc'),'pc_n'], 1,0))/sum((data['model']=='mcmc')),
             "em > naive": sum(np.where(data.loc[(data['model'] == 'em'), 'pc_m'] >= data.loc[(data['model'] == 'em'), 'pc_n'], 1,0))/sum((data['model']=='em')),
             "mcmc > em": sum(np.where(np.array(data.loc[(data['model'] == 'mcmc'), 'pc_m']) >= np.array(data.loc[(data['model'] == 'em'), 'pc_m']), 1,0))/sum((data['model']=='mcmc')),
             "mcmc > krip a":sum(np.where(data.loc[(data['model']=='mcmc'),'pc_m']>=data.loc[(data['model']=='mcmc'),'pc_aftr_prun_total'], 1,0))/sum((data['model']=='mcmc')),
             "em > krip a": sum(np.where(data.loc[(data['model'] == 'em'), 'pc_m'] >= data.loc[(data['model'] == 'em'), 'pc_aftr_prun_total'], 1, 0))/sum((data['model']=='em')),
             "krip a > naive": sum(np.where(data.loc[(data['model'] == 'em'), 'pc_aftr_prun_total'] >= data.loc[(data['model'] == 'em'), 'pc_n'], 1, 0))/sum((data['model']=='em')),
    }
    ## mcmc better than em on beta dists
    # sum(np.where(np.array(np.array(data.loc[(data['model'] == 'mcmc') & (
    #             (data['mode'] == 'beta3_2') | (data['mode'] == 'beta2_2') | (
    #                 data['mode'] == 'beta4_2')), 'pc_m'])) >= np.array(np.array(data.loc[(data['model'] == 'em') & (
    #             (data['mode'] == 'beta3_2') | (data['mode'] == 'beta2_2') | (data['mode'] == 'beta4_2')), 'pc_m'])), 1,
    #              0)) / sum((data['model'] == 'mcmc') & (
    #             (data['mode'] == 'beta3_2') | (data['mode'] == 'beta2_2') | (data['mode'] == 'beta4_2')))
    print(stats)
    size = 'small'
    # plot.saveplots()
    plot.plot_interactive()

