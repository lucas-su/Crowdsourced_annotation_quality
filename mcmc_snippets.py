
"""
        outcome = mean(tn[other])
        product of all trustworthiness scores for a given question for all cardinalities, summed for all questions that the evaluated tn is involved in

        """

        # y = np.log((math.gamma(1+t_n))/(math.gamma(1)*math.gamma(t_n)))+(t_n-1)*np.sum([
        #     np.log(1-(
        #         (t_n if k == n[1][f"q_{m}"] else ((1 - t_n) / self.cm)
        #          )
        #     ))])

        # y = np.prod([
        #         np.prod([
        #             np.prod([(t_n if k == n[1][f"q_{m}"] else ((1 - t_n) / self.cm))
        #             for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #         ])
        #         for k in range(self.K)])
        # for m in annotations.loc[(annotations['id_0']==n[1].ID)|
        #                          (annotations['id_1']==n[1].ID)|
        #                          (annotations['id_2']==n[1].ID)].index])


        # attempt with basically H = gamma(tn)/gamme(tn_new)
        # y= np.sum([
        #     np.sum([
        #          np.prod([
        #             (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"]) / self.cm)) * (1 / self.K) * n[1][
        #                 "T_model"]
        #             for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #          ])/sum([
        #             np.prod([
        #                 (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"]) / self.cm)) * (1 / self.K) * n[1][
        #                     "T_model"]
        #                 for n in user.loc[~np.isnan(user.loc[:, f"q_{m}"])].iterrows()
        #             ]
        #             ) for l in self.L
        #          ])
        #         for k in range(self.K)
        #     ])
        #     for m in annotations.loc[(annotations['id_0'] == n[1].ID) |
        #                              (annotations['id_1']==n[1].ID)|
        #                              (annotations['id_2']==n[1].ID)].index
        # ])


        # y = sum([
        #         sum([gamma.iloc[m, k] * (np.log(t_n) if k == n[1][f"q_{m}"] else (np.log(1 - t_n)/ self.cm))
        #
        #              for k in range(self.K)])
        #
        #         for m in self.M])

        # y = sum([n[1][f"q_{m}"]
        #          )
#
    # def gamma(self, k, user, m):
    #     """
    #     probability of true label for question m being equal to option k given the provided labels and the trustworthiness
    #     """
    #
    #
    #     # product of the modelled trustworthiness if answer is equal to k iterated over every user who answered question m
    #     # sum of all products of the modelled trustworthiness if answer is equal to answer option l iterated over every user who answered question m over all options l
    #
    #     """
    #     p(l_nm | GT_m == k, t_n) is implemented as (t_n if k==l_nm else (1-t_n)/cm)
    #     p(GT_m == k) and p(GT_m == l) implemented as (1/k)
    #     """
    #
    #     num = np.prod([
    #                     (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * n[1]["T_model"]
    #                    for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
    #                 ])
    #     denom = sum([
    #                 np.prod([
    #                         (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) * n[1]["T_model"]
    #                           for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
    #                 ]
    #                 ) for l in self.L])
    #     g = num/denom
    #     return g

    # @timeit
    # def e_step(self):
    #     for k in range(self.K):  # for each option
    #         with Pool(16) as p:
    #             result = p.map(partial(self.gamma,k, user), self.M)
    #         self.gamma_.loc[:,k] = result
    #     return self.gamma_

# def transition_model(self,x):
#     return np.random.normal(x, 0.3, (1,))[0]


# def prior(self,tn):
#     # returns 1 for all valid values of tn. Log(1) =0, so it does not affect the summation.
#     # returns 0 for all invalid values of tn (<=0 or >=1). Log(0)=-infinity, and Log(negative number) is undefined.
#     # It makes the new tn infinitely unlikely.
#     if (0<tn<1):
#         return 1
#     return np.spacing(0) # use spacing instead of 0 to prevent divide by zero

# def acceptance(self, x, x_new):
#     if x_new > x:
#         return True
#     else:
#         # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
#         # less likely x_new are less likely to be accepted
#         return (np.random.uniform(0, 1) < (np.exp(x_new - x)))


# def gamma(self, x, a, b):
#     return (math.gamma(a + b)*x**(a-1)*(1-x)**(b-1)) / (math.gamma(a) * math.gamma(b))


# def MH(self, user, annotations, n):
#     # n is a single row of user df
#
#     x_new = self.transition_model(n[1]['T_model'])
#     x_lik = self.p_D_theta(user, annotations, n, n[1]['T_model'])
#     x_new_lik = self.p_D_theta(user, annotations, n, x_new)
#     if self.acceptance(x_lik + np.log(self.prior(n[1]['T_model'])), x_new_lik + np.log(self.prior(x_new))):
#         return 1, x_new, n[1]['T_model']
#     else:
#         return 0, x_new, n[1]['T_model']


# # for MH, determine final modelled estimate
# for q in range(nQuestions):
#     k_w = np.zeros(car)
#     for k in range(car):
#
#         for d in range(dup):
#             if annotations.loc[q, f'annot_{d}'] == k:
#                 k_w = [k_w[i] + ((1 - user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
#                        k_w[i] + user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
#                 # k_w[k] += user.loc[annotations.loc[q, f'id_{d}'], 'T_model']
#             else:
#                 # k_w = [k_w[i]+((1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'])/(car-1)) if i!= k else k_w[i] for i in range(car)]
#                 k_w = [
#                     k_w[i] + ((user.loc[annotations.loc[q, f'id_{d}'], 'T_model']) / (car - 1)) if i != k else
#                     k_w[i] + 1-user.loc[annotations.loc[q, f'id_{d}'], 'T_model'] for i in range(car)]
#     annotations.loc[q, 'model'] = k_w.index(max(k_w))

# a = prop correct, b = total annotations


# T_beta = beta.pdf(results, a,b)*100/sum(beta.pdf(results, a,b))
# user.loc[:,'T_model']= (T_beta+np.spacing(3))/(1+np.spacing(3))
# user.loc[:, 'T_model'] = np.array(results)[:,0]
# a = sum(np.array(results)[:,1])
# b = sum(np.array(results)[:,2])
# accepted = []
# rejected = []
# acCols = []
# for n in range(user.__len__()):
#     acCols += [f'n_{n}A', f'n_{n}R']
# acceptionDF = pandas.DataFrame(columns=acCols)

# accepted = [result[1] if result[0]==1 else np.nan for result in results ]
# rejected = [result[1] if result[0]==0 else np.nan for result in results ]
# acceptionDF.loc[acceptionDF.__len__(), :] = np.array(list(zip(accepted,rejected))).flatten()
# user.loc[:, "T_model"] = results
# user.loc[:, f"t_weight_{i}"] = results