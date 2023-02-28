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

def debug(*args):
    print("DEBUG",*args)


import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import beta
plt.rcParams["figure.figsize"] = (10,15)


rng = default_rng()

nSamples = 1

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
        self.cardinality = len(alpha)
        self.gt = True if gt else False
        debug("Question constructor",name,alpha,gt,self.gt)
        self.value = np.zeros(self.prior.shape)
        if self.gt:
            self.value[self.gt] = nSamples
        else:
            for _ in range(nSamples):
                p = rng.dirichlet(self.prior)
                self.value += rng.multinomial(1,p)
            
        self.diff = difficulty
        self.annotations = []        # Keep track of the labels that were made for this question
        
    def addAnnotation(self,annot):
        self.annotations.append(annot)
        
    def computePosterior(self):
        if self.gt:
            # This is a special question, the ground-truth answer is known. Don't sample
            return
        
        alpha = np.array(self.prior)
        for l in self.annotations:
            for i in range(nSamples):
                t = l.annotator.sample()# sample trust
                alpha += t * l.value1ofk
        self.posterior = alpha
        debug("Sampling question posterior",self.name, self.posterior)
        
    def sample(self):
        """Sample the ground-truth value of this question"""
                    
        self.value = np.array(self.prior)
        p = rng.dirichlet(self.posterior)
        return p
        return rng.multinomial(1,p)
        
    def __repr__(self):
        if self.gt:
            return " +Question '%s',GT=%s" % (self.name,self.value)
        else:
            return "  Question '%s', V=%s" % (self.name,self.value)
        
    def plot(self,ax):
        ax.cla()
        p = self.sample()
        ax.bar(np.arange(self.cardinality),p)
        ax.set_title("Value %s" % (self.name))
        
        
class Annotator:
    """
    How trustworthy is an annotator
    """
    def __init__(self, name, a=1, b=1):
        self.name = name
        self.annotations = []
        self.prior = (a,b)
        self.posterior = (a,b)
    
    def addAnnotation(self, annot):
        self.annotations.append(annot)
    
    def computePosterior(self):
        alpha,beta = self.prior
        for a in self.annotations:
            for _ in range(nSamples):

                v = a.question.sample()
                t = self.sample()
                pos = t*v[a.value]# * np.prod(1.-np.concatenate((v[:a.value],v[(a.value+1):])))
                neg = (1.-t)/a.question.cardinality# * (1.-a.question.cardinality)**(a.question.cardinality-1)
#                debug(v,v[a.value],pos,neg)
                post = pos/(pos+neg)
                alpha += post
                beta += 1.-post
#                debug("trustworthiness ",self.name,a.question.name," a=",a.value,"q=",v, "post=",post,alpha,beta)
                    
        debug("Annotator posterior ",self.name,"a=",alpha,"b=",beta)
        self.posterior = (alpha,beta)
        
    def sample(self):
        """Sample the annotator's trustworthiness"""
        
        a,b = self.posterior
        return rng.beta(a,b)
    
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
    
    def __repr__(self):
        return "Annotation by %s of [%s]: %s" % (self.annotator.name, self.question, self.value)
        


# In[48]:


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

if __name__ == "__main__":

    cardinality = 2
    qAlpha = tuple((1e-5 for _ in range(cardinality)))

    data = [
        [ "Q1", "A", 0 ],
        [ "Q1", "B", 0 ],
    #    [ "Q1", "C", 0 ],
    #    [ "Q1", "D", 1 ],
        [ "Q2", "A", 1 ],
    #    [ "Q2", "B", 1 ],
    #    [ "Q2", "C", 1 ],
        [ "Q2", "D", 0 ],
        [ "Q3", "A", 0 ],
        [ "Q3", "B", 0 ],
        [ "Q3", "C", 0 ],
    #    [ "Q3", "D", 1 ],
    #    [ "Q4", "A", 1 ],
        [ "Q4", "B", 1 ],
    #    [ "Q4", "C", 1 ],
        [ "Q4", "D", 0 ]
    ]


    names = set()
    qnames = set()
    for d in data:
        names.add(d[1])
        qnames.add(d[0])

    annotators = {name: Annotator(name,1e-1,1e-5) for name in names }
    questions = {name: Question(name,qAlpha) for name in qnames }
    debug(questions)

    annotations = []

    for d in data:
        annotations.append(Annotation(annotators[d[1]], questions[d[0]], d[2]))

    for n,a in annotators.items():
        print(a)

    fig,axs = plt.subplots(4,2)
    for it in range(50):
        for n,q in questions.items():
            q.computePosterior()
        for n,a in annotators.items():
            a.computePosterior()
    i=0
    for n,q in questions.items():
        q.plot(axs.flat[i])
        i+=1
    for n,a in annotators.items():
        a.plot(axs.flat[i])
        i+=1
    plt.show()

        
    print('done')

# In[ ]:




