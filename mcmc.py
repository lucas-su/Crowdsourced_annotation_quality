
import platform
from collections import Counter
import numpy as np
import pandas, pickle
from multiprocessing import Pool
from functools import partial
from scipy.stats import beta
from datetime import datetime
import os
from create_simulation_data import createData
from settings import *
from scipy.special import logsumexp

from numpy.random import default_rng
rng = default_rng()

from functools import wraps
from time import time
from majority_vote import majority

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts} sec')
          
        return result
    return wrap

def debug_print(*args):
    if debug:
        print(f'Debug:', *args)


class Question:
    """
    We take into account:
    - how many different answers are possible for a particular question,
    - What the actual answer is (potentially)
    - How difficult a question is (for later? Can we take into account the possibility that an
    annotator believes to know what the right answer is separately from the probability that they
    give a different answer?)
    """
    def __init__(self, id, KG, GT, car, difficulty=None):
        self.id = id        
        self.prior = np.array([priors['qAlpha'] for _ in range(car)])
         
        self.posterior = np.array([priors['qAlpha'] for _ in range(car)])
        self.basePrior = np.array([priors['qAlpha'] for _ in range(car)])
        self.cardinality = len(self.prior)
        self.KG = KG
        self.GT = GT
        self.C = 0 # start at 0 to prevent sampling from annealing in first iteration
        self.car = car
        self.diff = difficulty
        self.annotations = []        # Keep track of the labels that were made for this question
        self.postsamples = []
        
    def addAnnotation(self,annot):
        self.annotations.append(annot)
        
    def computePosterior(self, nSamples, u_idx = None):
        if self.KG:
            # This is a special question, the ground-truth answer is known. Don't sample
            return 10000 * np.eye(self.cardinality)[self.GT] + np.array(self.prior)
        elif np.any([annotation.annotator.KG for annotation in self.annotations]):
            # is we know at least one of the annotators is good, we don't need to sample
            # this should really be: if we have a KG, take the answer from the person, but this is the same as taking GT
            return 10000 * np.eye(self.cardinality)[self.GT] + np.array(self.prior)
        else:
            alpha = np.array(self.prior)
            debug_print(f'sample question self.post: {self.posterior}')
            for i in range(nSamples):

                if u_idx == None:
                    ls = self.annotations
                else:
                    # select labels for annotators who are included (some might be excluded in the warmup)
                    ls = [l for l in self.annotations if l.annotator.id in u_idx]
                ts = [l.annotator.sample() for l in ls] # trustworthinesses
                ts = [np.spacing(3) if t ==0 else 1-np.spacing(3) if t ==1 else t for t in ts]
                a = np.ones(self.cardinality)

                for v in range(self.cardinality):
                    # For every possible answer
                    cs = [l.value == v for l in ls]  # check which annotator is correct for this answer
                    for t, c in zip(ts, cs):
                        a[v] *= t if c else ((1. - t)/(self.car-1))  # Compute the probability that this combination of correctnesses happens
                    # debug("posterior question", self, "for v=", v, ":", ts, cs)
                # debug(" --> a", a)
                # alpha += len(self.annotations) * ((a / a.sum()) / nSamples)
                with np.errstate(all="raise"):
                    try:
                        alpha += len(ls) * ((a / a.sum()) / nSamples)
                    except:
                        pass

            #
            #     for l in self.annotations:
            #         t = l.annotator.sample()# sample trust
            #         # if (t<1e-10):
            #         #     t = np.spacing(3)
            #         # if (1-t) < 1e-10:
            #         #     t = 1.-np.spacing(3)
            #         s = t * l.value1ofk + ((1 - t) / (self.cardinality - 1)) * (1. - l.value1ofk)
            #         # if s.argmax() != self.GT:
            #         #     print(f's -> {s} t -> {t} l -> {l.value1ofk} GT -> {self.GT}')
            #         a += np.log(s)
            #         # a *= s
            #     #     debug(l, "t=",t,"s=",s,"a=",a)
            #     # debug(" ==> a=",a/a.sum())
            #     with np.errstate(all="raise"):
            #         try:
            #             alpha += len(self.annotations) * (np.exp(a-logsumexp(a))/nSamples)
            #         except:
            #             pass

            return alpha
            
    def sample(self):
        """Sample the ground-truth value of this question"""
        if rng.uniform()<self.C:
            p = rng.uniform(size=self.car)
            return p/p.sum()
        else:
            return rng.dirichlet(self.posterior)

    def best(self):
        return self.posterior.argmax()

    def logProb(self):
        debug_print("logprob q: ", self.posterior)
        return np.log(self.posterior.max())-np.log(np.sum(self.posterior))

    def __repr__(self):
        if self.gt:
            return "Q+'%s',%s" % (self.name,self.posterior)
        else:
            return "Q '%s',%s" % (self.name,self.posterior)
        
    def plot(self,ax):
        ax.cla()
        p = 0.
        for _ in range(10):
            p += self.sample()
        p /= 10.
        ax.bar(np.arange(self.cardinality),p)
        ax.set_title("Value %s" % (self.name))
        
        
class Annotator:
    """
    How trustworthy is an annotator
    """
    def __init__(self, id, type, T_given, car):
        self.id = id
        self.T = T_given
        self.KG = True if type == 'KG' else False
        self.annotations = []
        self.basePrior = np.array((priors['aAlpha'],priors['aBeta']))
        self.prior = np.array((priors['aAlpha'],priors['aBeta']))
        self.posterior = np.array((priors['aAlpha'],priors['aBeta']))
        self.postsamples = []
        self.C = 0 # start at 0 to prevent sampling from annealing in first iter
        self.car = car
        # self.annealdist = np.array(priors['anneal'])

    def addAnnotation(self, annot):
        self.annotations.append(annot)

    def computePosterior(self, nSamples, q_idx = None):
        if self.KG:
            # return self.annotations.__len__()+priors['aAlpha'],priors['aBeta']
            return 1000, 1e-10
        else:
            alpha,beta = self.prior
            if q_idx == None:
                ls = self.annotations
            else:
                ls = [l for l in self.annotations if l.annotator.id in q_idx]

            for a in ls:
                debug_print(f'sample annot self.post: {self.posterior}')
                for _ in range(nSamples):
                    v = a.question.sample()

                    # t = self.sample() # of current posterior
                    #
                    # if (t<1e-10):
                    #     t = 1e-10
                    # if t > 1.-1e-10:
                    #     t = 1-1e-10
                    chance = (1. - v[a.value]) / (self.car - 1)
                    pos = v[a.value]
                    # pos = v[a.value] - chance # picking an extremely unlikely constitutes negative evidence, means that v[a] may not be exactly 0
                    neg = chance
                    post = pos/(pos+neg) # v value -chance/ v value

                    # debug(a,v,"pos=%0.2g,neg=%0.2g" % (pos,neg))
                    alpha += (post)/nSamples
                    beta += ((1.-post))/nSamples
                    # alp = v[a.value]/nSamples
                    # alpha += alp
                    # bet = (1-v[a.value])/(nSamples*(self.car-1))
                    # beta += bet

                    # debug("trustworthiness ",self.name,a.question.name,"a=",a.value,"q=",v, "t=",t,"post=",post,alpha,beta)

            # debug("Annotator posterior ",self.name,"a=",alpha,"b=",beta, "num of annot=",len(self.annotations))
            if alpha <= 0:

                alpha = np.spacing(0)
            if beta <= 0:
                beta = np.spacing(0)

            return alpha, beta
        
    def sample(self):
        """Sample the annotator's trustworthiness"""
        if rng.uniform()<self.C:
            return 0. if rng.uniform() < 0.5 else 1.

        else:
            a,b = self.posterior
            return rng.beta(a,b)


    def logProb(self):

        return np.log(self.posterior.max())-np.log(np.sum(self.posterior))
    
    def __repr__(self):
        s = "Annotator(%s)" % self.id
        for a in self.annotations:
            s += "\n    -> %s" % a.__repr__()
        return s
    
    def plot(self,ax):
#        ax.cla()
        x = np.linspace(0,1,100)
        a,b = self.posterior
        y = beta.pdf(x,a,b)
        ax.plot(x,y)
        ax.set_title("Trust %s" % (self.id))
        

class Annotation:
    def __init__(self, annotator, question, value):
        self.annotator = annotator
        annotator.addAnnotation(self)
        self.question = question
        question.addAnnotation(self)
        self.value = value
        self.value1ofk = np.eye(question.prior.__len__())[value]
        # debug("Annotation constructor",self.value,self.value1ofk)
        
        
    def isCorrect(self):
        return self.value == self.question.value
        
    def sampleValue(self):
        """For data generation, generate annotations"""
        pass

    def logProb(self):
        x = self.question.posterior / self.question.posterior.sum()
        return np.log(x[self.value])
    
    def __repr__(self):
        return "Annot by %s:%s of [%s]" % (self.annotator.name, self.value, self.question)
        

class mcmc():
    # @timeit
    def __init__(self, car, items, user):
        self.M = np.arange(0,nQuestions)    # Questions
        self.L = np.arange(0,car)             # Given label per question
        self.car = car                          # Number of answer options
        self.cm = car-1                       # -1 because there's one good answer and the rest is wrong
        self.iter = 0

        self.annotators = {id: Annotator(id, type, T_given, car) for id, type, T_given in zip(user.ID, user.type, user.T_given)}
        self.questions = {qid: Question(qid, KG, GT, car) for qid, KG, GT in zip(items.ID, items.KG, items.GT)}

        self.annotations = []


        for row in items.iterrows():
            for i in range(dup):
                self.annotations.append(Annotation(self.annotators[row[1][f'id_{i}']], self.questions[row[0]], row[1][f'annot_{i}']))
    
    def sampleQIteration(self,nSamples, i):
        posts = self.questions[i].computePosterior(nSamples)
        return posts
    
    def sampleAIteration(self,nSamples, i):
        return self.annotators[i].computePosterior(nSamples)

    def anneal(self,n, warmup):
        if n + 5 > warmup:
            C = 0
        else:
            # C = 0.3 / (1.5 ** n)
            C = 0

        # for q in self.questions.values():
        #     q.C = C
        for a in self.annotators.values():
            a.C = C

    # @timeit
    def modelEvidence(self):
        logEvidenceQ = 0
        logEvidenceA = 0
        for q in self.questions.values():
            logEvidenceQ = logEvidenceQ + q.logProb()
        for a in self.annotators.values():
            # logEvidenceA = logsumexp((logEvidenceA,a.logProb()))
            logEvidenceA = logEvidenceA + a.logProb()
        return logEvidenceQ, logEvidenceA        

    # @timeit
    def process_pc_posterior(self):

        # average modelled trustworthiness over selected samples
        for u in self.annotators.values():
            # u.T = np.mean(u.postsamples, axis=0).max()
            u.T_model = rng.beta(*np.mean(u.postsamples, axis=0),1)

        for q in self.questions.values():
            alphas = q.postsamples

            # p = np.mean([rng.dirichlet(alphas[i]) for i in range(keep_n_samples)], axis=0)
            # q.model = p.argmax()

            p = np.mean(alphas, axis=0)
            q.model = p.argmax()

    def KG_warmup(self, a_idx, i_idx):
        i_from_a = []

        # compute posteriors for current annotators based on current items
        # determine items associated with annotators
        for a in self.annotators.values():
            if a.id in a_idx:
                a.posterior = a.computePosterior(nSamples, i_idx)
                i_from_a += [ann.question.id for ann in a.annotations]

        # compute posteriors for items based on current annotators
        # compute posteriors for items associated with annotators_ based on current annotators
        # determine annotators associated with items
        a_from_i = []
        for i in self.questions.values():
            if i.id in i_idx or i.id in i_from_a:
                i.posterior = i.computePosterior(nSamples, a_idx)
                a_from_i += [ann.annotator.id for ann in i.annotations]

        # compute posteriors for annotators _associated with items_ based on current items
        for a in self.annotators.values():
            if a.id in a_from_i:
                a.posterior = a.computePosterior(nSamples, i_idx)

        # merge current annotators with annotators associated with items
        a_idx.update(a_from_i)
        # merge current items with items associated with annotators
        i_idx.update(i_from_a)

        return a_idx, i_idx


    # @timeit
    def run(self, keep_n_samples, car, nQuestions, users, items, priors, nSamples):
        
        # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
        posteriorindices = (warmup * [False])+[x % sample_interval == 0 for x in range(keep_n_samples*sample_interval)]
        sample_cnt = 0
        


        a_idx = set(users.loc[(users['type'] == 'KG'), 'ID'])
        i_idx = set(items.loc[(items['KG'] == True), 'ID'])
        for a in self.annotators.values():
            if a.id in a_idx:
                i_idx.update([ann.question.id for ann in a.annotations])

        while a_idx.__len__() != users.__len__() or i_idx.__len__() != items.__len__():
            a_len = a_idx.__len__()
            i_len = i_idx.__len__()
            a_idx, i_idx = self.KG_warmup(a_idx, i_idx)
            if a_len == a_idx.__len__() and i_len == i_idx.__len__():
                if a_len+i_len >0:
                    print('Island detected during warmup')
                break
            # print(f'warmed up {a_idx.__len__()} users out of {users.__len__()}')
            # print(f'warmed up {i_idx.__len__()} questions out of {items.__len__()}')

        debug_print('done warming up')
        post_history_cols = np.concatenate((np.array(
            [[f'q_{i}', f'q_{i}_correct'] for i in range(nQuestions)]).flatten(), np.array(
            [[f'a_{i}', f'a_{i}_dT'] for i in range(nAnnotator)]).flatten()))
        history = pandas.DataFrame(columns=post_history_cols)
        with Pool(ncpu) as p:
        
            while self.iter < posteriorindices.__len__():
        
                ## sample l_hat
                # first only the KG's, as that primes the lhats for the other samples with the right bias
                indices = items.loc[(items['KG'] == True), 'ID']
                if indices.__len__()>0:
                    
                    results = p.map(partial(self.sampleQIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        debug_print(f'Question posterior KG after sampling:{res}')
                        self.questions[i].posterior = np.array(res)

                ## sample tn
                # first do the KG users
                indices = users.loc[(users['type'] == 'KG'), 'ID']
                if indices.__len__()>0:
                    results = p.map(partial(self.sampleAIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        debug_print(f'Annot posterior KG: {res}')
                        self.annotators[i].posterior = np.array(res)
                    # no need to sample known good users: they are known good and therefore T = 1
                
                # after the KG tn's, do the rest of the lhats
                indices = items.loc[(items['KG'] == False), 'ID']
                
                results = p.map(partial(self.sampleQIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    debug_print(f'Question posterior normal:{res}')
                    self.questions[i].posterior = np.array(res)
                    
                
                # finally the rest of tn's 
                indices = users.loc[(users['type'] != 'KG'), 'ID']
                results = p.map(partial(self.sampleAIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    debug_print(f'Annot posterior normal: {res}')
                    self.annotators[i].posterior = np.array(res)
                
                if posteriorindices[self.iter]:
                    for annotator in self.annotators.values():
                        annotator.postsamples.append(annotator.posterior)
                    for question in self.questions.values():
                        question.postsamples.append(question.posterior)

                    sample_cnt += 1
                self.iter += 1
                self.anneal(self.iter, warmup)

                history.loc[self.iter] =  np.concatenate(( np.array([[q.posterior, np.argmax(q.posterior) == q.GT] for q in self.questions.values()],dtype=object).flatten(), np.array([[a.posterior, abs(rng.beta(*a.posterior) - a.T)] for a in self.annotators.values()],dtype=object).flatten()))

        assert(sample_cnt == keep_n_samples)

        self.process_pc_posterior()
        
        self.pc_m = np.sum([q.model==q.GT for _,q in self.questions.items()])/nQuestions
        # self.pc_n = np.sum([q.GT==maj_ans for (_,q), maj_ans in zip(self.questions.items(), majority(annotations, nQuestions, car))])/nQuestions
        return history




class ModelSel:
    def __init__(self,keep_n_samples, car, nQuestions, user, annotations, priors,nModels,nSamples):
        models = []
        best_history = None
        bestEvidence = -np.inf
        bestModel = None
        n =0
        stop_iteration = nModels + 7

        while n < nModels:
            models.append(mcmc(car, annotations, user))
            m = models[-1]

            history = m.run(keep_n_samples, car, nQuestions, user, annotations, priors, nSamples)

            leQ, leA = m.modelEvidence()
            le = leQ + leA

            p_DgivenTheta = (le+np.log(1/car))-le
            if np.exp(p_DgivenTheta) < 0.5:
                print(f'Evidence low: lhat -> {np.exp(leQ)} tn -> {np.exp(leA)} pc_m -> {m.pc_m}')
                # print(f'previous pc_m was: {m.pc_m}')
                nModels += 1

            if le>bestEvidence:
                # print(f'new best evidence {np.exp(le)} is better than old evidence {np.exp(bestEvidence)}')
                bestEvidence = le
                bestModel = m
                best_history = history
                self.bestQ = leQ
                self.bestA = leA

                if le ==1: # cannot get better than 1
                    break

            if n >= stop_iteration:
                # give up finding better model
                break

            n+=1
        with open(f'{session_dir}/output/post_hist_{T_dist}.pickle', 'wb') as file:
            pickle.dump(best_history, file)
        self.model = bestModel

    def best(self):
        return self.model
    
if __name__ == "__main__":
    for size in datasetsize_list:
        for sweeptype, T_dist_list in sweeps.items():
            for car in car_list:
                for dup in dup_list:
                    priors = set_priors()
                    for p_fo in p_fo_list:
                        for kg_q in kg_q_list:
                            for kg_u in kg_u_list:
                                mcmc_data = pandas.DataFrame(
                                    columns=['size', 'iterations', 'car', 'T_dist', 'sweeptype', 'dup', 'p_fo', 'kg_q', 'kg_u',
                                             'mcmc', 'pc_m', 'pc_n', 'pc_n_KG', 'CertaintyQ', 'CertaintyA'])

                                session_dir = set_session_dir(size, sweeptype, car, dup, p_fo, kg_q, kg_u) + f'session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
                                for T_dist in T_dist_list:
                                    os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)

                                    createData(f'{session_dir}', car, T_dist, dup, p_fo, kg_u, ncpu, size)
                                    print(f"Datasetsize {size}, cardinality {car}, distribution {T_dist}, annotations per item {dup}, malicious {p_fo}, known good items {kg_q}, known good users {kg_u}")
                                    for keep_n_samples in keep_samples_list:
                                        # open dataset for selected parameters
                                        with open(f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_user.pickle',
                                                  'rb') as file:
                                            users = pickle.load(file)
                                        with open(
                                                f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{kg_u}_annotations_empty.pickle',
                                                'rb') as file:
                                            items = pickle.load(file)

                                        ## add parameters to dataframes
                                        # known goods
                                        items[f'KG'] = np.zeros(items.__len__())
                                        if kg_q>0:
                                            items.loc[:kg_q - 1, f'KG'] = 1
                                        # annotations[f'KG'] = [np.random.choice([0,1], p=[1 - kg_q, kg_q]) for _ in range(annotations.__len__())]

                                        # global nQuestions
                                        nQuestions = items.__len__()
                                        # majority(annotations, nQuestions, car)
                                        maj = majority(items, nQuestions, car, dup, users)
                                        sel_model = ModelSel(keep_n_samples, car, nQuestions, users, items, priors, nModels, nSamples)
                                        # confs = sorted([(i, np.exp(j.logProb())) for i,j in enumerate(sel_model.model.questions.values())], key=lambda x:x[1])
                                        confs = np.array([np.exp(j.logProb()) for j in sel_model.model.questions.values()])
                                        confindices = np.where(confs<0.5)
                                        # if confindices[0].__len__()>0:
                                        #     print(f'Questions {confindices[0]} need extra attention: confidences are: {[confs[i] for i in confindices[0]]}')
                                        # else:
                                        #     print('No further answering needed')

                                        # [(i, conf), axis=1) for i, conf in enumerate(np.exp(prob)) for prob in sel_model.model.questions.values()


                                        t_diff = np.mean([abs(a.T_model-a.T) for a in sel_model.model.annotators.values()])
                                        print(f'conf: {np.exp(sel_model.bestQ+sel_model.bestA)} \n'
                                              f'prop. correct modelled: \t\t\t\t {sel_model.model.pc_m} \n'
                                              f'prop. correct maj. vote: \t\t\t\t {maj.pc}\n'
                                              f'prop. correct maj. vote KG corrected:\t {maj.pc_KG}\n'
                                              f'avg. diff. T: {t_diff}')

                                        # create mcmc_data dataframe
                                        mcmc_data.loc[mcmc_data.__len__(), :] = [size, keep_n_samples, car, T_dist, sweeptype, dup, p_fo, kg_q, kg_u, sel_model, sel_model.model.pc_m, maj.pc, maj.pc_KG, sel_model.bestQ, sel_model.bestA]

                                        # save data
                                        with open(f'{session_dir}/output/mcmc_annotations_T_dist-{T_dist}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                            pickle.dump(items, file)
                                        with open(f'{session_dir}/output/mcmc_user_T_dist-{T_dist}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                            pickle.dump(users, file)
                                        with open(f'{session_dir}/output/mcmc_data.pickle', 'wb') as file:
                                            pickle.dump(mcmc_data, file)