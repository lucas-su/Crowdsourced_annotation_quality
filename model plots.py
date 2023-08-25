import os.path
import pickle
# from settings import *
import matplotlib.pyplot as plt
import create_stats
import pandas
from matplotlib import cm
import numpy as np
from em import EM
from mcmc import *
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.axes3d import get_test_data
from create_stats import find_params


class plots():
    # def __init__(self):
    #     self.figcar, self.axscar = plt.subplots(dup_list.__len__(), p_fo_list.__len__(), sharex=True, sharey=True)
    #     self.figuerror, self.axsuerror = plt.subplots(dup_list.__len__(), p_fo_list.__len__(), sharex=True, sharey=True)
    #     self.figsave, self.axssave = plt.subplots()
    #     self.figsave.set_size_inches(8,4)
    #     self.figsave.subplots_adjust(top=0.98, bottom=0.12,right=0.83,left=0.08)

    def plot_one(self, mode, dup, p_fo, kg_q, iterations, datasetsize, kg_u):
        self.axssave.plot(car_list, data.loc[
            (data['size'] == datasetsize) &
            (data['model'] == 'mcmc') &
            (data['iterations'] == iterations['mcmc']) &
            pandas.Series([item in car_list for item in data['car']]) &
            (data['mode'] == mode) &
            (data['p_fo'] == p_fo) &
            (data['kg_q'] == kg_q) &
            (data['kg_u'] == kg_u) &
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
            (data['kg_q'] == kg_q) &
            (data['kg_u'] == kg_u) &
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
            (data['kg_q'] == kg_q) &
            (data['kg_u'] == kg_u) &
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
            (data['kg_q'] == kg_q) &
            (data['kg_u'] == kg_u) &
            (data['dup'] == dup),
            'pc_aftr_prun_total'  # pc_aftr_prun_total pc_aftr_prun
        ], label='Krip. α')
        self.axssave.set_xlabel('Cardinality')
        self.axssave.set_xticks([3,5,7])
        self.axssave.legend(loc='lower left', bbox_to_anchor=(1, 0, 1, 1))
        self.axssave.set_ylabel(f'Proportion correct')

    def saveplots(self):
        # mode, dup, p_fo, kg_q, iterations, size, kg_u
        plotdata = [['beta2_4', 2,0,0,iterations,'small', 0],
                    ['beta2_4', 9,0,0,iterations,'small', 0],
                    ['beta4_2', 2, 0, .2, iterations, 'comb', .2],
                    ['beta4_2', 2, 0, .1, iterations, 'large', .1],
                    ['beta2_2', 2, 0, .2, iterations, 'comb', .2],
                    ['beta2_2', 2, 0.2, .2, iterations, 'comb', .2]]
        for plotdatum in plotdata:
            self.plot_one(*plotdatum)
            plt.savefig(f'C:\\Users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\PC_mode-{plotdatum[0]}_dup-{plotdatum[1]}_p_fo-{plotdatum[2]}_kg_q-{plotdatum[3]}_size-{plotdatum[5]}_kg_u{plotdatum[6]}.png', dpi=300)
            self.axssave.clear()

    def plot(self):
        # todo makemake non-global
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                    (data['kg_q'] == kg_q) &
                    (data['kg_u'] == kg_u) &
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
                valmax= kg_q_list[-1],
                valinit= val,
                valstep= kg_q_list,
                orientation="vertical"
            )
            return pkg_slider
        pkg_slider = getKGSlider(kg_q_list[0])

        def updatepkg(val):
            global kg_q
            global pkg_slider
            kg_q = val
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
                valmax= kg_u_list[-1],
                valinit= val,
                valstep= kg_u_list,
                orientation="vertical"
            )
            return pkgu_slider
        pkgu_slider = getKGUSlider(kg_u_list[0])

        def updatepkgu(val):
            global kg_u
            global pkgu_slider
            kg_u = val
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

    def plot_delta_T(self, sweeptype, car, dup, p_fo, kg_q, kg_u):
        self.figdelta_T, self.axsdelta_T = plt.subplots(figsize=(6,3.5))
        x = np.arange(11) / 10

        deltaT_mcmc = data.loc[(data['model'] == 'mcmc'), 'T_diff']
        deltaT_em =  data.loc[(data['model'] == 'em'), 'T_diff']
        self.axsdelta_T.plot(x, deltaT_mcmc, label='mcmc')
        self.axsdelta_T.plot(x, deltaT_em, label='em')
        # plt.show()


        self.axsdelta_T.legend()
        plt.savefig(f'plots/legend/delta_T_datasetsize_{size}-dist_{sweeptype}-car_{car}-dup_{dup}-p_fo_{p_fo}-kg_q_{kg_q}-kg_u_{kg_u}.png')
        plt.savefig(f'{session_dir}/delta_T_plot.png')
        # plt.show()
        plt.close()


    def plot_pc_T(self, datasetsize, sweeptype, car, dup, p_fo, kg_q, kg_u):
        self.figpc_T, self.axspc_T = plt.subplots(figsize=(6,3.5))
        mcmcpc_m = data.loc[(data['model'] == 'mcmc'), 'pc_m']
        mcmc_sd = data.loc[(data['model'] == 'mcmc'), 'pc_m_SD']

        mcmcpc_m_cert = data.loc[(data['model'] == 'mcmc'), 'pc_m_cert']


        certQ = data.loc[(data['model'] == 'mcmc'), 'CertaintyQ']
        certA = data.loc[(data['model'] == 'mcmc'), 'CertaintyA']

        empc_m = data.loc[(data['model'] == 'em'), 'pc_m']
        em_sd = data.loc[(data['model'] == 'em'), 'pc_m_SD']

        naivepc = data.loc[(data['model'] == 'mcmc'), 'pc_n']
        naive_sd = data.loc[(data['model'] == 'mcmc'), 'pc_n_SD']

        naiveKGpc = data.loc[(data['model'] == 'mcmc'), 'pc_n_KG']
        naiveKG_sd = data.loc[(data['model'] == 'mcmc'), 'pc_n_KG_SD']

        pc_krip = data.loc[(data['model'] == 'mcmc'), 'pc_krip']
        pc_krip_SD = data.loc[(data['model'] == 'mcmc'), 'pc_krip_SD']

        x = np.arange(11)/10

        # cardinality
        self.axspc_T.hlines(1/car, x[0], x[-1], label='1/cardinality', colors='#2ca02c')

        # maj. vote
        self.axspc_T.plot(x, naivepc, label='Maj. vote', color='firebrick')
        self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(naivepc + naive_sd, dtype=float)], [max(sd, 0) for sd in np.array(naivepc - naive_sd, dtype=float)],color='firebrick', alpha=0.2 )

        # KG maj. vote
        if kg_q > 0 or kg_u > 0:
            self.axspc_T.plot(x, naiveKGpc, label='Maj. vote KG', color='purple')
            self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(naiveKGpc+naiveKG_sd, dtype=float)], [max(sd, 0) for sd in np.array(naiveKGpc-naiveKG_sd, dtype=float)], color='purple', alpha=0.2)


        # em
        self.axspc_T.plot(x, empc_m, label='EM', color='darkorange')
        self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(empc_m+em_sd, dtype=float)], [max(sd, 0) for sd in np.array(empc_m-em_sd, dtype=float)],color='darkorange', alpha=0.2 )

        # mcmc
        self.axspc_T.plot(x, mcmcpc_m, label='MCMC', color='#1f77b4')
        self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(mcmcpc_m+mcmc_sd, dtype=float)], [max(sd, 0) for sd in np.array(mcmcpc_m-mcmc_sd, dtype=float)], color='#1f77b4', alpha=0.2)
        # self.axspc_T.plot(x, mcmcpc_m_cert, label='PC. MCMC cert', color='black')

        # krip
        # self.axspc_T.plot(x, pc_krip, label='krip', color='gold')
        # self.axspc_T.fill_between(x, [min(sd, 1) for sd in np.array(pc_krip + pc_krip_SD, dtype=float)], [max(sd, 0) for sd in np.array(pc_krip - pc_krip_SD, dtype=float)],color='gold', alpha=0.2 )



        # certainty
        certQ = [np.exp(cert) for cert in certQ]
        certA = [np.exp(cert) for cert in certA]
        # self.axspc_T.plot(x, certQ,  label='conf. Question', color='#d62728', alpha=0.5, linestyle='dashed')
        # self.axspc_T.plot(x, certA,  label='conf. Annotator', color='#9467bd', alpha=0.5, linestyle='dashed')

        # if sweeptype == 'propT':
        #     self.axspc_T.set_xlabel('Proportion T=1 vs. T=0')
        # elif sweeptype == 'beta':
        #     self.axspc_T.set_xlabel('Mean density in beta distribution')
        # elif sweeptype == 'beta_small':
        #     self.axspc_T.set_xlabel('Mean density in beta distribution')
        # else:
        #     raise ValueError

        # self.axspc_T.set_ylabel('— Proportion item labels correct\n- - - Confidence in questions and answers')
        # self.axspc_T.set_title(f'Prop. of items correct for car {car}, duplication factor {dup}, known good items {kg_q}, datasetsize {datasetsize}, known good users {kg_u}')
        os.makedirs(f'plots/no_legend', exist_ok=True)
        os.makedirs(f'plots/legend', exist_ok=True)
        plt.savefig(f'plots/no_legend/datasetsize_{size}-dist_{sweeptype}-car_{car}-dup_{dup}-p_fo_{p_fo}-kg_q_{kg_q}-kg_u_{kg_u}.png')
        self.axspc_T.legend()
        plt.savefig(f'plots/legend/datasetsize_{size}-dist_{sweeptype}-car_{car}-dup_{dup}-p_fo_{p_fo}-kg_q_{kg_q}-kg_u_{kg_u}.png')
        plt.savefig(f'{session_dir}/plot.png')
        # plt.show()
        plt.close()

if __name__ == "__main__":
    # model = "mcmc"  # options "em" or "mcmc"
    latexpath = f'C:\\users\\admin\\pacof\\notes\\Papers\\trustworthiness modelling\\figures\\em_mcmc_plots\\'

    beta_smin = 0.01
    beta_smax = 0.8
    beta_min = 1
    beta_max = 6
    sweeps = {'beta': [f'beta{round(flt, 2)}_{round(beta_max - flt, 2)}' for flt in
                             np.linspace(beta_min, beta_max - beta_min, 11)],
              'beta_small': [f'beta2{round(flt, 2)}_{round(beta_smax - flt, 2)}' for flt in
                             np.linspace(beta_smin, beta_smax - beta_smin, 11)],
              "propT": [f'propT_{round(flt, 2)}' for flt in np.arange(0, 1.1, 0.1)]}

    iterations = {'em':10,
                  'mcmc': 100}
    plot = plots()
    walk = os.walk('sessions')
    for step in walk:
        session_dir = ""
        try:
            if step[1][0][:7] == 'session':
                session_dir = step[0]
        except:
            continue
        if session_dir != "":
            create_stats.main(session_dir, step, sweeps)

            # if not os.path.exists(f'{session_dir}/plot.png'):
            with open(f'{session_dir}/stats.pickle', 'rb') as file:
                data = pickle.load(file)
            data = data.loc[data['session'] == 'avg']
            size, sweeptype, car, dup, p_fo, kg_q, kg_u = find_params(session_dir)
            plot.plot_pc_T(size, sweeptype, car, dup, p_fo, kg_q, kg_u)
            plot.plot_delta_T(sweeptype, car, dup, p_fo, kg_q, kg_u)



