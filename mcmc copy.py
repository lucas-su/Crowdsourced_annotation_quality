
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
        self.prior = np.array([priors['aAlpha'] for _ in range(car)])
         
        self.posterior = np.array([priors['aAlpha'] for _ in range(car)])
        self.basePrior = np.array([priors['aAlpha'] for _ in range(car)])
        self.cardinality = len(self.prior)
        self.KG = KG
        self.GT = GT
            
        self.diff = difficulty
        self.annotations = []        # Keep track of the labels that were made for this question
        self.postsamples = []
        
    def addAnnotation(self,annot):
        self.annotations.append(annot)
        
    def computePosterior(self, nSamples):
        if self.KG:
            # This is a special question, the ground-truth answer is known. Don't sample
            return np.eye(self.cardinality)[self.GT]
        elif np.any([annotation.annotator.KG for annotation in self.annotations]):
            # is we know at least one of the annotators is good, we don't need to sample
            # this should really be: if we have a KG, take the answer from the person, but this is the same as taking GT
            return np.eye(self.cardinality)[self.GT]
        else:
            alpha = np.array(self.prior)
            for i in range(nSamples):
                a = np.ones(alpha.shape)
                for l in self.annotations:
                    t = l.annotator.sample()# sample trust
                    if (t<1e-10):
                        t = 1e-10
                    if (1-t) < 1e-10:
                        t = 1.-1e-10
                    s = t * l.value1ofk + (1-t)/(self.cardinality-1) * (1.-l.value1ofk)
                    a *= s
                #     debug(l, "t=",t,"s=",s,"a=",a)
                # debug(" ==> a=",a/a.sum())
                alpha += (a/a.sum())/nSamples
                    # alpha += t/nSamples * l.value1ofk + (1-t)/(nSamples*self.cardinality**2) * (1.-l.value1ofk)
            return alpha
         
            
            
    def sample(self):
        """Sample the ground-truth value of this question"""
                    
#        self.value = np.array(self.prior)
        p = rng.dirichlet(self.posterior)
        return p
#        return rng.multinomial(1,p)

    def best(self):
        return self.posterior.argmax()

    def logProb(self):
        return np.log(self.posterior.max()/self.posterior.sum())

    def anneal(self,n):
        self.prior = self.prior / (2. ** n)
    
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
    def __init__(self, id, type):
        self.id = id
        self.KG = True if type == 'KG' else False
        self.annotations = []
        self.basePrior = (priors['aAlpha'],priors['aAlpha'])
        self.prior = (priors['aAlpha'],priors['aAlpha'])
        self.posterior = (priors['aAlpha'],priors['aAlpha'])
        self.postsamples = []
    
    def addAnnotation(self, annot):
        self.annotations.append(annot)

    def computePosterior(self, nSamples):
        if self.KG:
            return (1,np.spacing(0))
        else:
            alpha,beta = self.prior
            for a in self.annotations:
                for _ in range(nSamples):
                    v = a.question.sample()
                    t = self.sample() # of current posterior
                    if (t<1e-10):
                        t = 1e-10
                    if t > 1.-1e-10:
                        t = 1-1e-10
                    chance = 1.-v[a.value]
                    pos = np.max([v[a.value]-chance,0.])
                    neg = chance
                    post = pos/(pos+neg)
                    # debug(a,v,"pos=%0.2g,neg=%0.2g" % (pos,neg))
                    alpha += (post)/nSamples                    
                    beta += (1.-post)/nSamples

                    # debug("trustworthiness ",self.name,a.question.name,"a=",a.value,"q=",v, "t=",t,"post=",post,alpha,beta)
                        
            # debug("Annotator posterior ",self.name,"a=",alpha,"b=",beta, "num of annot=",len(self.annotations))
            return (alpha,beta)
        
    def sample(self):
        """Sample the annotator's trustworthiness"""
        
        a,b = self.posterior
        return rng.beta(a,b)

    def anneal(self,n):
        a,b = self.basePrior
        self.prior = (a/(2**n),b/(2**n))
    
    def __repr__(self):
        s = "Annotator(%s)" % self.name
        for a in self.annotations:
            s += "\n    -> %s" % a.__repr__()
        return s
    
    def plot(self,ax):
#        ax.cla()
        x = np.linspace(0,1,100)
        a,b = self.posterior
        y = beta.pdf(x,a,b)
        ax.plot(x,y)
        ax.set_title("Trust %s" % (self.name))
        

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

        self.annotators = {id: Annotator(id, type) for id, type in zip(user.ID, user.type)}
        self.questions = {qid: Question(qid, KG, GT, car) for qid, KG, GT in zip(annotations.ID, annotations.KG, annotations.GT)}

        self.annotations = []

        for row in annotations.iterrows():
            for i in range(car):
                self.annotations.append(Annotation(self.annotators[row[1][f'id_{i}']], self.questions[row[0]], row[1][f'annot_{i}']))
    
    def sampleQIteration(self,nSamples, i):
        return self.questions[i].computePosterior(nSamples)
        
    
    def sampleAIteration(self,nSamples, i):
        return self.annotators[i].computePosterior(nSamples)

    def anneal(self,n):
        for _,q in self.questions.items():
            q.anneal(n)
        for _,a in self.annotators.items():
            a.anneal(n)
            
    def modelEvidence(self):
        logEvidence = 0
        for _,q in self.questions.items():
            logEvidence += q.logProb()
        return logEvidence        
    
    @timeit
    def process_pc_posterior(self):

        # average modelled trustworthiness over selected samples
        for _, u in self.annotators.items():
            u.T = rng.beta(*np.mean(u.postsamples, axis=0),1)

        for _,q in self.questions.items():
            alphas = q.postsamples

            p = np.mean([rng.dirichlet(alphas[i]) for i in range(keep_n_samples)], axis=0)
            q.model = p.argmax()
        
    
    def run(self, keep_n_samples, car, nQuestions, user, annotations, priors, nSamples):

        # generate binary array of to be selected estimates for posterior: ten rounds warmup, then every third estimate
        posteriorindices = (warmup * [False])+[x % sample_interval == 0 for x in range(keep_n_samples*sample_interval)]

        # counter to keep track of how many samples are taken
        sample_cnt = 0

        with Pool(ncpu) as p:
            while self.iter < posteriorindices.__len__():
                if self.iter % 10 == 0:
                    print("iteration: ", self.iter)

                ## sample l_hat
                # first only the KG's, as that primes the lhats for the other samples with the right bias
                indices = annotations.loc[(annotations['KG']==True), 'ID']
                if indices.__len__()>0:
                    
                    results = p.map(partial(self.sampleQIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        self.annotations[i].posterior = res

                ## sample tn
                # first do the KG users
                indices = user.loc[(user['type']=='KG'), 'ID']
                if indices.__len__()>0:
                    results = p.map(partial(self.sampleAIteration, nSamples), indices)
                    for i, res in zip(indices, results):
                        self.annotators[i].posterior = res
                    # no need to sample known good users: they are known good and therefore T = 1
                                
                # after the KG tn's, do the rest of the lhats
                indices = annotations.loc[(annotations['KG'] == False), 'ID']
                
                results = p.map(partial(self.sampleQIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    self.annotations[i].posterior = res
                
                # finally the rest of tn's 
                indices = user.loc[(user['type'] != 'KG'), 'ID']
                results = p.map(partial(self.sampleAIteration, nSamples), indices)
                for i, res in zip(indices, results):
                    self.annotators[i].posterior = res
                
                if posteriorindices[self.iter]:
                    for _, annotator in self.annotators.items():
                        annotator.postsamples.append(annotator.posterior)
                    for _, question in self.questions.items():
                        question.postsamples.append(question.posterior)

                    sample_cnt += 1
                self.iter += 1
                self.anneal(0.5*self.iter)
            
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
        for n in range(nModels):
            models.append(mcmc(car, annotations, user))
            m = models[-1]
            m.run(keep_n_samples, car, nQuestions, user, annotations, priors, nSamples)
            
            le = m.modelEvidence()
            if le>bestEvidence:
                bestEvidence = le
                bestModel = m
        self.model = bestModel

    def best(self):
        return self.model
    

if __name__ == "__main__":
    
    if platform.system() == 'Windows':
        ncpu = 16
    else:
        ncpu = 32
    ## settings

    if platform.system() == 'Windows': # for quick debug
        warmup = 3
        nSamples = 1
        sample_interval = 1
        keep_samples_list = [5]
    else:
        warmup = 10
        nSamples = 5
        # keep a sample every sample_interval iterations
        sample_interval = 3
        # n samples to keep
        keep_samples_list = [20]

    session_dir = f'sessions/prior-{priors["aAlpha"]}_{priors["aBeta"]}-car{car_list[0]}/session_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    os.makedirs(f'{os.getcwd()}/{session_dir}/output', exist_ok=True)

    resume_mode = False
    if resume_mode:
        with open(f'sessions/mcmc_data_{"_".join(T_dist_list)}.pickle', 'rb') as file:
            mcmc_data = pickle.load(file)
    else:
        mcmc_data = pandas.DataFrame(
            columns=['size', 'iterations', 'car', 'mode', 'dup', 'p_fo', 'p_kg', 'p_kg_u', 'mcmc', 'pc_m', 'pc_n'])

    for size in datasetsize_list: 
        for car in car_list:
            for T_dist in T_dist_list:
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
                                    annotations[f'KG'] = [np.random.choice([0,1], p=[1-p_kg,p_kg]) for _ in range(annotations.__len__())]

                                    # global nQuestions
                                    nQuestions = annotations.__len__()

                                    sel_model = ModelSel(keep_n_samples, car, nQuestions, user, annotations, priors, nModels,nSamples)

                                    # create mcmc_data dataframe
                                    mcmc_data.loc[mcmc_data.__len__(), :] = [size, keep_n_samples, car, T_dist, dup, p_fo, p_kg, p_kg_u, None, 0, 0]
                                    # init mcmc object
                                    mcmc_data.loc[(mcmc_data['size'].values == size) &
                                                  (mcmc_data['iterations'].values == keep_n_samples) &
                                                  (mcmc_data['car'].values == car) &
                                                  (mcmc_data['mode'].values == T_dist) &
                                                  (mcmc_data['dup'].values == dup) &
                                                  (mcmc_data['p_fo'].values == p_fo) &
                                                  (mcmc_data['p_kg'].values == p_kg) &
                                                  (mcmc_data['p_kg_u'].values == p_kg_u), ['mcmc', 'pc_m', 'pc_n']] = [sel_model, sel_model.model.pc_m, sel_model.model.pc_n]
 

                                    # save data
                                    with open(f'{session_dir}/output/mcmc_annotations_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(annotations, file)
                                    with open(f'{session_dir}/output/mcmc_user_data_size-{size}_mode-{T_dist}_dup-{dup}_car-{car}_p-fo-{p_fo}_p-kg-{p_kg}_p-kg-u{p_kg_u}_iters-{keep_n_samples}.pickle', 'wb') as file:
                                        pickle.dump(user, file)
                                    with open(f'{session_dir}/output/mcmc_data_size-{size}{"_".join(T_dist_list)}.pickle', 'wb') as file:
                                        pickle.dump(mcmc_data, file)