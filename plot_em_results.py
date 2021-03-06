import matplotlib.pyplot as plt
import pickle, pandas
import numpy as np
import scipy.stats as stats

if __name__ == '__main__':
    with open('simulation data/users_with_scores.pickle', 'rb') as file:
        users = pickle.load(file)
    with open('simulation data/meta.pickle', 'rb') as file:
        meta = pickle.load(file)

    steps = 1000
    nIters = (users.columns.__len__() - 3)
    x = np.linspace(0,2 , steps)
    fig = plt.figure()
    # plt.rcParams['text.usetex'] = True
    frame = fig.add_subplot(111)
    frame.spines['top'].set_visible(False)
    frame.spines['right'].set_visible(False)
    frame.spines['bottom'].set_visible(False)
    frame.spines['left'].set_visible(False)
    frame.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax = fig.subplots(nIters, 2)
    included_ids = users.loc[users.included == 1, 'id']
    for it in range(0,nIters):
        ax[it][0].scatter(users.quality, users.loc[:,f"q_weight_{it}"])
        ax[it][1].plot(x, stats.norm.pdf(x, meta[f'mean1_{it}'], meta[f'var1_{it}']))
        ax[it][1].plot(x, stats.norm.pdf(x, meta[f'mean2_{it}'], meta[f'var2_{it}']))
    #     np.linspace(0, 10, users.__len__()), [users.loc[id, 'quality'] if id in included_ids else np.nan for id in users.id] , 'o')
    # ax.plot(np.linspace(0, 10, users.__len__()), [np.nan if id in included_ids else users.loc[id, 'quality'] for id in users.id], 'o', color='red')

    # plt.savefig(f"plots/removals all iters norm_{norm_method} remove_{remove_method} sd_thresh_{sd_threshold}.png")
    ax[-1][0].set_xlabel('Per user latent annotator quality score')
    ax[-1][1].set_xlabel('Distributions over normalized annotator agreement score')
    frame.set_ylabel('p(x=g_{high})')
    plt.show()

    # fig, ax = plt.subplots(nIters, 1)
    # for i in range(nIters):
    #     removed = users.loc[users.included == i * -1, 'id']
    #     ax[i].plot(np.linspace(0, 10, users.__len__()), [users.loc[id, f'q_weight_{i}'] if id in included_ids else np.nan for id in users.id] , 'o')
    #     # ax[i].plot(np.linspace(0, 10, users.__len__()), [np.nan if id in included_ids else users.loc[id, f'q_weight_{i}'] for id in users.id], 'o', color='red')
    #     ax[i].plot(np.linspace(0, 10, users.__len__()),
    #                [users.loc[id, f'q_weight_{i}'] if id in removed else np.nan for id in users.id], 'o',
    #                color='red')
    #
    # # plt.savefig(f"plots/removals per iteration norm_{norm_method} remove_{remove_method} sd_thresh_{sd_threshold}.png")
    # plt.show()
    #
    # _, ax = plt.subplots(1, 1)
