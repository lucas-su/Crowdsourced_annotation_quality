
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

from numpy.random import default_rng
rng = default_rng()

from functools import wraps
from time import time

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
        self.C = 1 # 'temperature' of annealing 
        self.car = car
        self.diff = difficulty
        self.annotations = []        # Keep track of the labels that were made for this question
        self.postsamples = []
        
    def addAnnotation(self,annot):
        self.annotations.append(annot)
        
    def computePosterior(self, nSamples):
        if self.KG:
            # This is a special question, the ground-truth answer is known. Don't sample
            return np.eye(self.cardinality)[self.GT]+np.spacing(0)
        elif np.any([annotation.annotator.KG for annotation in self.annotations]):
            # is we know at least one of the annotators is good, we don't need to sample
            # this should really be: if we have a KG, take the answer from the person, but this is the same as taking GT
            return np.eye(self.cardinality)[self.GT]+np.spacing(0)
        else:
            alpha = np.array(self.prior)
            debug_print(f'sample question self.post: {self.posterior}')
            for i in range(nSamples):
                # a = np.ones(alpha.shape)
                a = np.zeros(alpha.shape)
                for l in self.annotations:
                    t = l.annotator.sample()# sample trust
                    if (t<1e-10):
                        t = 1e-10
                    if (1-t) < 1e-10:
                        t = 1.-1e-10
                    s = t * l.value1ofk + ((1 - t) / (self.cardinality - 1)) * (1. - l.value1ofk)
                    a += np.log(s)
                    # a *= s
                #     debug(l, "t=",t,"s=",s,"a=",a)
                # debug(" ==> a=",a/a.sum())
                alpha += ((np.exp(a)/np.exp(a).sum())/nSamples)
                # alpha += ((a / a.sum()) / nSamples)
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
        return np.log(self.posterior.max()/self.posterior.sum())

    def anneal(self,n):
        self.C = 1/(1.5**n)

    
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
        self.C = 1
        self.car = car
        # self.annealdist = np.array(priors['anneal'])

    def addAnnotation(self, annot):
        self.annotations.append(annot)

    def computePosterior(self, nSamples):
        if self.KG:
            return (10,np.spacing(0))
        else:
            alpha,beta = self.prior
            for a in self.annotations:
                debug_print(f'sample annot self.post: {self.posterior}')
                for _ in range(nSamples):
                    v = a.question.sample()
                    # t = self.sample() # of current posterior
                    #
                    # if (t<1e-10):
                    #     t = 1e-10
                    # if t > 1.-1e-10:
                    #     t = 1-1e-10
                    chance = 1.-v[a.value]
                    pos = np.max([v[a.value]-chance,0.])
                    neg = chance
                    post = pos/(pos+neg)
                    # debug(a,v,"pos=%0.2g,neg=%0.2g" % (pos,neg))
                    alpha += (post)/nSamples
                    beta += ((1.-post))/nSamples

                    # debug("trustworthiness ",self.name,a.question.name,"a=",a.value,"q=",v, "t=",t,"post=",post,alpha,beta)
                        
            # debug("Annotator posterior ",self.name,"a=",alpha,"b=",beta, "num of annot=",len(self.annotations))
            return alpha,beta
        
    def sample(self):
        """Sample the annotator's trustworthiness"""
        if rng.uniform()< self.C:
            return rng.uniform()
        else:
            a,b = self.posterior
            return rng.beta(a,b)

    def anneal(self,n):
        self.C = 1/(1.5**n)

    
    def logProb(self):
        return np.log(self.posterior.max()/self.posterior.sum())
    
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
    
    def __init__(self, car, annotations, user):
        self.M = np.arange(0,nQuestions)    # Questions
        self.L = np.arange(0,car)             # Given label per question
        self.K = car                          # Number of answer options
        self.cm = car-1                       # -1 because there's one good answer and the rest is wrong
        self.iter = 0

        self.annotators = {id: Annotator(id, type, T_given, car) for id, type, T_given in zip(user.ID, user.type, user.T_given)}
        self.questions = {qid: Question(qid, KG, GT, car) for qid, KG, GT in zip(annotations.ID, annotations.KG, annotations.GT)}

        self.annotations = []

        for row in annotations.iterrows():
            for i in range(dup):
                self.annotations.append(Annotation(self.annotators[row[1][f'id_{i}']], self.questions[row[0]], row[1][f'annot_{i}']))
    
    def sampleQIteration(self,nSamples, i):
        posts = self.questions[i].computePosterior(nSamples)
        return posts
    
    def sampleAIteration(self,nSamples, i):
        return self.annotators[i].computePosterior(nSamples)

    def anneal(self,n):
        for _,q in self.questions.items():
            q.anneal(n)
        for _,a in self.annotators.items():
            a.anneal(n)
            
    def modelEvidence(self):
        for _,q in self.questions.items():
            logEvidenceQ = q.logProb()
        for _,a in self.annotators.items():
            logEvidenceA = a.logProb()
        return logEvidenceQ, logEvidenceA        
    
    def process_pc_posterior(self):

        # average modelled trustworthiness over selected samples
        for u in self.annotators.values():
            # u.T = np.mean(u.postsamples, axis=0).max()
            u.T_model = rng.beta(*np.mean(u.postsamples, axis=0),1)

        for q in self.questions.values():
            alphas = q.postsamples

            # p = np.mean([rng.dirichlet(alphas[i]) for i in range(keep_n_samples)], axis=0)
            # q.model = p.argmax()

            p = rng.dirichlet(np.mean(alphas, axis=0))
            q.model = p.argmax()
        
    # @timeit
    def run(self, keep_n_samples, car, nQuestions, user, annotations, priors, nSamples):
        
        # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
        posteriorindices = (warmup * [False])+[x % sample_interval == 0 for x in range(keep_n_samples*sample_interval)]
        sample_cnt = 0
        
        # counter to keep track of how many samples are taken
        
        with Pool(ncpu) as p:
        
            while self.iter < posteriorindices.__len__():
        
                ## sample l_hat
                # first only the KG's, as that primes the lhats for the other samples with the right bias
                indices = annotations.loc[(annotations['KG']==True), 'ID']
                if indices.__len__()>0:
                    
                    results = p.map(partial(self.sampleQIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        debug_print(f'Question posterior KG after sampling:{res}')
                        self.questions[i].posterior = np.array(res)
                        

                ## sample tn
                # first do the KG users
                indices = user.loc[(user['type']=='KG'), 'ID']
                if indices.__len__()>0:
                    results = p.map(partial(self.sampleAIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        debug_print(f'Annot posterior KG: {res}')
                        self.annotators[i].posterior = np.array(res)
                    # no need to sample known good users: they are known good and therefore T = 1
                
                # after the KG tn's, do the rest of the lhats
                indices = annotations.loc[(annotations['KG'] == False), 'ID']
                
                results = p.map(partial(self.sampleQIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    debug_print(f'Question posterior normal:{res}')
                    self.questions[i].posterior = np.array(res)
                    
                
                # finally the rest of tn's 
                indices = user.loc[(user['type'] != 'KG'), 'ID']
                results = p.map(partial(self.sampleAIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    debug_print(f'Annot posterior normal: {res}')
                    self.annotators[i].posterior = np.array(res)
                
                if posteriorindices[self.iter]:
                    for _, annotator in self.annotators.items():
                        annotator.postsamples.append(annotator.posterior)
                    for _, question in self.questions.items():
                        question.postsamples.append(question.posterior)

                    sample_cnt += 1
                self.iter += 1
                self.anneal(self.iter)

             
        assert(sample_cnt == keep_n_samples)

        self.process_pc_posterior()
        
        self.pc_m = np.sum([q.model==q.GT for _,q in self.questions.items()])/nQuestions
        self.pc_n = np.sum([q.GT==maj_ans for (_,q), maj_ans in zip(self.questions.items(), majority(annotations, nQuestions, car))])/nQuestions
        


def majority(annotations, nQuestions, car):
    maj_ans =[]
    for q in range(nQuestions):
        # weights for all k options list
        k_w = []
        for k in range(car):
            # counter for number of people who chose option k
            d_w = 0
            for d in range(dup):
                if annotations.at[q, f'annot_{d}'] == k:
                    d_w += 1
            k_w.append(d_w)
        max_val = max(k_w)
        max_indices = []
        for i, k in enumerate(k_w):
            if k == max_val:
                max_indices.append(i)
        maj_ans.append(max_indices[np.random.randint(max_indices.__len__())])
    return maj_ans

class ModelSel:
    def __init__(self,keep_n_samples, car, nQuestions, user, annotations, priors,nModels,nSamples):
        models = []
        bestEvidence = -np.inf
        bestModel = None
        n =0
        while n < nModels:
            models.append(mcmc(car, annotations, user))
            m = models[-1]
            m.run(keep_n_samples, car, nQuestions, user, annotations, priors, nSamples)
            
            leQ, leA = m.modelEvidence()
            le = leQ + leA
            if np.exp(le) < 0.5:
                print(f'Evidence low: lhat -> {np.exp(leQ)} tn -> {np.exp(leA)} pc_m -> {m.pc_m}')
                # print(f'previous pc_m was: {m.pc_m}')
                nModels += 1

            if le>bestEvidence:
                # print(f'new best evidence {np.exp(le)} is better than old evidence {np.exp(bestEvidence)}')
                bestEvidence = le
                bestModel = m
                self.bestQ = leQ
                self.bestA = leA

                if le ==1: # cannot get better than 1
                    break

            if n >= nModels + 10:
                # give up finding better model
                break

            n+=1

        self.model = bestModel

    def best(self):
        return self.model
    
if __name__ == "__main__":

    mcmc_data = pandas.DataFrame(columns=['size', 'iterations', 'car', 'T_dist', 'dup', 'p_fo', 'p_kg', 'p_kg_u',
                                          'mcmc', 'pc_m', 'pc_n', 'CertaintyQ', 'CertaintyA'])
    for size in datasetsize_list:
        for car in car_list:
            priors = set_priors(size, priors, car)
            for T_dist in T_dist_list:
                session_dir = f'sessions/datasetsize_{size}/cardinality_{car}/prior-{priors["aAlpha"]}_{priors["aBeta"]}/session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

                os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)

                createData(f'{session_dir}', car, T_dist, dup_list, p_fo_list, p_kg_u_list, ncpu, size)

                for keep_n_samples in keep_samples_list:
                    for dup in dup_list:
                        for p_fo in p_fo_list:
                            for p_kg in p_kg_list:
                                for p_kg_u in p_kg_u_list:
                                    # open dataset for selected parameters
                                    with open(f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_user.pickle',
                                              'rb') as file:
                                        user = pickle.load(file)
                                    with open(
                                            f'{session_dir}/simulation data/{T_dist}/pickle/{size}_{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-u-{p_kg_u}_annotations_empty.pickle',
                                            'rb') as file:
                                        annotations = pickle.load(file)

                                    ## add parameters to dataframes
                                    # known goods
                                    if p_kg > 0:
                                        while not annotations[f'KG'].any(): # ensure that there is at least one KG
                                            annotations[f'KG'] = [np.random.choice([0,1], p=[1-p_kg,p_kg]) for _ in range(annotations.__len__())]
                                    else:
                                        annotations[f'KG'] = np.zeros(annotations.__len__())
                                    # global nQuestions
                                    nQuestions = annotations.__len__()

                                    sel_model = ModelSel(keep_n_samples, car, nQuestions, user, annotations, priors, nModels,nSamples)
                                    # confs = sorted([(i, np.exp(j.logProb())) for i,j in enumerate(sel_model.model.questions.values())], key=lambda x:x[1])
                                    confs = np.array([np.exp(j.logProb()) for j in sel_model.model.questions.values()])
                                    confindices = np.where(confs<0.5)
                                    # if confindices[0].__len__()>0:
                                    #     print(f'Questions {confindices[0]} need extra attention: confidences are: {[confs[i] for i in confindices[0]]}')
                                    # else:
                                    #     print('No further answering needed')

                                        # [(i, conf), axis=1) for i, conf in enumerate(np.exp(prob)) for prob in sel_model.model.questions.values()
                                        
                                    t_diff = np.mean([abs(a.T_model-a.T) for a in sel_model.model.annotators.values()])
                                    print(f'conf: {np.exp(sel_model.bestQ+sel_model.bestA)}, pc_m: {sel_model.model.pc_m}, pc_n: {sel_model.model.pc_n}, t_diff: {t_diff}')

                                    # create mcmc_data dataframe
                                    mcmc_data.loc[mcmc_data.__len__(), :] = [size, keep_n_samples, car, T_dist, dup, p_fo, p_kg, p_kg_u, sel_model, sel_model.model.pc_m, sel_model.model.pc_n, sel_model.bestQ, sel_model.bestA]
 
                                    # save data
                                    with open(f'{session_dir}/output/mcmc_annotations_data_size-{size}_T_dist-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(annotations, file)
                                    with open(f'{session_dir}/output/mcmc_user_data_size-{size}_T_dist-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(user, file)
                                    with open(f'{session_dir}/output/mcmc_data_size-{size}{"_".join(T_dist_list)}.pickle', 'wb') as file:
                                        pickle.dump(mcmc_data, file)