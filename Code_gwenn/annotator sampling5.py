#!/usr/bin/env python
# coding: utf-8

# # Distributions
#  
# $
# \newcommand{\q}{\mathbf{q}}
# \newcommand{\l}{\mathbf{l}}
# \newcommand{\p}{\mathbf{p}}
# \newcommand{\betadist}{\mathrm{Beta}}
# \newcommand{\multidist}{\mathrm{Mult}}
# \newcommand{\catdist}{\mathrm{Cat}}
# \newcommand{\dirdist}{\mathrm{Dir}}
# \newcommand{\dist}{\sim}
# \newcommand{\boldmu}{\boldsymbol{\mu}}
# \newcommand{\boldalpha}{\boldsymbol{\alpha}}
# $
#  
# Consider a set of annotatators $\{a_n\}_1^N$, which can each have a level of trustworthiness $t_n$ (the porobability that they will answer correctly) and a propensity for different answers if they do not answer correctly $\mathbf{p}_n$. The probability of label $\mathbf{l}_{nm}$, a one-of-k encoding of the label by annotator $a_n$ to question $\mathbf{q}_m$ is then given by the mixture model:
# $$
# p(\mathbf{l}_{nm}) = t_n \q_m + (1-t_n) \p_n
# $$
# 
# where 
# $$
# t_n \dist \betadist(\alpha_n,\beta_n)\\
# \q_m \dist \catdist(\boldmu_m)\\
# \boldmu_m \dist \dirdist(\boldalpha_\mu)\\
# \p_n \dist \dirdist(\boldalpha_p)
# $$
# 
# So, the posterior parameter values for the different distributions can be obtained by sampling
# 
# $$
# t_n | \{\l_{nm}, \q_{nm}) \} \dist \betadist(\alpha,\beta)
# $$
# where $\alpha = \alpha_n + \sum_m \l_{nm}\q_m$ and $\beta = xxx$
# 
# $$
# \q_m | \{t_n, \l_{nm} \}) \dist \catdist(\boldmu)
# $$
# where $\boldmu \dist \dirdist(\alpha_k = \sum_n t_n l_{nm}^k)$ 
# 
# In a first version, it would be logical for $\p_n \dist 1$
# 
# 
# 
# <!-- Here we have a choice: either we consider actively malicious annotators, or we do not. Let us start by considering that annotators can be actively malicious, so that $t=1$ means that the annotator always annotates a question correctly, and $t=0$ means that that annotator never answers the question correctly. How the annotator selects their answer if they get it wrong is also open to modelling: for example, it could be uniform, or an annotator might consistently select the first answer. So, let's see:
# $$
# a_m \approx  1\\
# t_n \approx Beta(\alpha_n, \beta_n)\\
# l_{nm} \approx \left\{ 
# \begin{array}
# a_m & \text{ with probability } t_n \\
# Categorical(\theta_n) & \text{Otherwise}\\
# \end{array}
# \right.
# $$-->

# In[1]:


def debug(*args):
    pass
def info(*args):
    print("INFO",*args)
def error(*args):
    print("ERROR",*args)    


# In[6]:


# def debug(*args):
#     print("DEBUG",*args)


# In[19]:


import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import beta
plt.rcParams["figure.figsize"] = (10,15)


rng = default_rng()

nSamples = 3

class Question:
    """
    We take into account:
    - how many different answers are possible for a particular question,
    - What the actual answer is (potentially)
    - How difficult a question is (for later? Can we take into account the possibility that an
    annotator believes to know what the right answer is separately from the probability that they
    give a different answer?)
    """
    def __init__(self, name, alpha, gt=None, difficulty=None):
        self.name = name        
        self.prior = np.array(alpha)
        self.posterior = self.prior.copy()
        self.basePrior = self.prior.copy()
        self.cardinality = len(alpha)
        self.gt = True if gt else False
#        self.value = np.zeros(self.prior.shape)
#        if self.gt:
#            self.value[self.gt] = nSamples
#        else:
#            for _ in range(nSamples):
#                p = rng.dirichlet(self.prior)
#                self.value += rng.multinomial(1,p)
            
        self.diff = difficulty
        self.annotations = []        # Keep track of the labels that were made for this question
        
    def addAnnotation(self,annot):
        self.annotations.append(annot)
        
    def computePosterior(self):
        if self.gt:
            # This is a special question, the ground-truth answer is known. Don't sample
            return

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
                debug(l, "t=",t,"s=",s,"a=",a)
            debug(" ==> a=",a/a.sum())
            alpha += (a/a.sum())/nSamples
                # alpha += t/nSamples * l.value1ofk + (1-t)/(nSamples*self.cardinality**2) * (1.-l.value1ofk)
        self.posterior = alpha
        debug("Sampling question posterior",self.name, self.posterior)
        
        
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
    def __init__(self, name, a=1, b=1):
        self.name = name
        self.annotations = []
        self.basePrior = (a,b)
        self.prior = (a,b)
        self.posterior = (a,b)
    
    def addAnnotation(self, annot):
        self.annotations.append(annot)
    
    def computePosterior(self):
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
                debug(a,v,"pos=%0.2g,neg=%0.2g" % (pos,neg))
                alpha += (post)/nSamples                    
                beta += (1.-post)/nSamples

                # debug("trustworthiness ",self.name,a.question.name,"a=",a.value,"q=",v, "t=",t,"post=",post,alpha,beta)
                    
        debug("Annotator posterior ",self.name,"a=",alpha,"b=",beta, "num of annot=",len(self.annotations))
        self.posterior = (alpha,beta)
        
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
        self.value1ofk = np.zeros(len(question.prior))
        self.value1ofk[value] = 1
        debug("Annotation constructor",self.value,self.value1ofk)
        
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
        


class Model:
    def __init__(self,data, qAlpha):
        names = set()
        qnames = set()
        for d in data:
            names.add(d[1])
            qnames.add(d[0])

        self.annotators = {name: Annotator(name,1e2,1.e1) for name in names }
        self.questions = {name: Question(name,qAlpha) for name in qnames }
        # debug(questions)

        self.annotations = []

        for d in data:
            self.annotations.append(Annotation(self.annotators[d[1]], self.questions[d[0]], d[2]))

    def sampleIteration(self):
        for n,q in self.questions.items():
            q.computePosterior()
        for n,a in self.annotators.items():
            a.computePosterior()

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


class ModelSel:
    def __init__(self,data,qAlpha,N=10,numIter=5):
        models = []
        bestEvidence = -np.inf
        bestModel = None
        for n in range(N):
            models.append(Model(data,qAlpha))
            m = models[-1]
            for i in range(numIter):
                m.sampleIteration()
                m.anneal(i)
            
            le = m.modelEvidence()
            debug("Model",n,"logEvidence:",le)
            if le>bestEvidence:
                bestEvidence = le
                bestModel = m
        self.model = bestModel

    def best(self):
        return self.model
        
    
# In[3]:


"""
GT:
Question 1 = 2
Question 2 = 0
Question 3 = 0

trustworthiness
A:  1
B:  1
C: Always first
"""

# cardinality = 2
# qAlpha = tuple((1e-5 for _ in range(cardinality)))

# data = [
#     [ "Q1", "A", 0 ],
#     [ "Q1", "B", 0 ],
# #    [ "Q1", "C", 0 ],
# #    [ "Q1", "D", 1 ],
#     [ "Q2", "A", 1 ],
# #    [ "Q2", "B", 1 ],
# #    [ "Q2", "C", 1 ],
#     [ "Q2", "D", 0 ],
#     [ "Q3", "A", 0 ],
#     [ "Q3", "B", 0 ],
#     [ "Q3", "C", 0 ],
# #    [ "Q3", "D", 1 ],
# #    [ "Q4", "A", 1 ],
#     [ "Q4", "B", 1 ],
# #    [ "Q4", "C", 1 ],
#     [ "Q4", "D", 0 ],
# ]


# # In[17]:


t = .8
card = 5
apc = 6

trust = {
    "A":t,
    "B":t,
    "C":t,
    "D":t,
    "E":t,
    "F":t,
    "G":t,
    "H":t,
    "I":t,
    "J":t,
}


def genData(numQuestions,card,trust,apc):
    data = []
    gt = {}
    names = list(trust.keys())
    for i in range(numQuestions):
        v = rng.integers(card)# ground truth
        qName = "Q%d[%d]" %(i,v)
        qAnnot = set()
        gt[qName] = v
        for _ in range(apc): #Annotators per question
            l = rng.uniform()
            a = rng.integers(len(trust)) # which annotator
            while a in qAnnot:
                a = rng.integers(len(trust)) # Avoid repeat annotations                
            qAnnot.add(a)
            data.append([qName, names[a],0])
            if l < trust[names[a]]: # correct answer
                data[-1][2] = v
            else:
                data[-1][2] = rng.integers(card)
                while data[-1][2] == v:
                    data[-1][2] = rng.integers(card)
                
            
    return data,gt, tuple((1.e-5 for _ in range(card)))

data,gt,qAlpha = genData(200,card,trust,apc)
# print(gt)
# for d in data:
#     print(d,gt[d[0]])

def majority(data):
    cnt = {}    
    for x in data:
        name = x[0]
        v = x[2]

        if name not in cnt:
            cnt[name] = {}
            
        if v in cnt[name]:
            cnt[name][v] += 1
        else:
            cnt[name][v] = 1

    res = {}
    for n,vals in cnt.items():
        max = 0
        maxv = 0
        for v,c in vals.items():
            if c>max:
                max = c
                maxv = v
        res[n]=maxv
    return res


def eval(pred, gt):
    good = 0
    bad = 0
    for n,v in pred.items():
        if v == gt[n]:
            good+=1
        else:
            bad +=1
    return good,bad,good/(good+bad)



# In[20]:
np.set_printoptions(precision=2)


print("Majority vote:",eval(majority(data),gt))
sel = ModelSel(data,qAlpha,20,10)
m = sel.best()

res = {}
for n,q in m.questions.items():
    res[n] = q.best()
print("   -> Model:",eval(res,gt))
    

    

# names = set()
# qnames = set()
# for d in data:
#     names.add(d[1])
#     qnames.add(d[0])

# annotators = {name: Annotator(name,1e-0,1.e-1) for name in names }
# questions = {name: Question(name,qAlpha) for name in qnames }
# debug(questions)

# annotations = []

# for d in data:
#     annotations.append(Annotation(annotators[d[1]], questions[d[0]], d[2]))

# for n,a in annotators.items():
#     print(a)

# fig,axs = plt.subplots(4,4)
# for it in range(20):
#     # info("Iteration",it)
#     for n,q in questions.items():
#         q.computePosterior()
#     for n,a in annotators.items():
#         a.computePosterior()


# i=0
# for n,a in annotators.items():
#     a.plot(axs.flat[i])
#     i+=1

# for n,q in questions.items():
#    if i==16:
#         break
#    q.plot(axs.flat[i])
#    i+=1        
# plt.show()
        

# print("Majority vote:",eval(majority(data),gt))
# res = {}
# for n,q in questions.items():
#     res[n] = q.best()
# print("Model:",eval(res,gt))
   



# %%
print(data)

# %%
