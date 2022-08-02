"""
"""
"""
# Model of annotations

## Basic ideas

1. Annotators have some intrinsic reliability. Some will be very accurate over many questions, others will not. We assume that this does not change over time, and that the reliability can be expressed as a probability of correct answer.
1. Questions have different inherent difficulty, and the ground truth answers to the questions (datapoints) is not known. We assume that each question has exactly one correct answer; for categorical questions, we assume that a question has a right answer that can be given with a certain probability, for continuous answers, we assume that the annotators will give an answer that is normally distributed around the correct answer with an annotator-specific standard deviation
1. We do not assume that annotators are malicious or biased. If the annotator does not give the right answer to a categorical question, we assume that any of the other answers can be given with uniform probability.

The idea is to learn the parameters of the model by expectation maximisation. In the E-step, we optimise the "responsibilities" $\gamma_{qa}$, the probability that question $q$ has ground truth answer $a$; in the M-step, we learn the annotator-specific model parameters.

$\newcommand{\label}{{l}}$
$\newcommand{\Labels}{\mathcal{L}}$
$\newcommand{\ground}{{\hat{l}}}$
$\newcommand{\Ground}{\mathcal{\hat L}}$
$\newcommand{\Trust}{\mathcal{T}}$

The full model is as follows. Let $a_n, 1\le n \le N$ be the set of annotators and $\ground_m, 0 \le m \le M$ be the ground truth values for the datapoints (questions). Annotators annotate multiple questions, and a single question can be annotated by multiple annotators. Let annotation (label) $\label_{nm}$ be the annotation given by annotator $n$ to question $m$ and $\L_n = \{ l_{nk} \}$ the set annotations from annotator $n$. Finally, we consider that each annotator $n$ has a certain level of trustworthiness $t_n \in \Trust$

We then define the complete joint probability of the annotations, ground truths and annotator trustworthiness as follows:

$$
p(\Ground,\Labels,\Trust) = \prod_{n=1}^N \prod_{m \in \Labels_n} p(\label_{nm} \mid \ground_m,t_n) \, p(\ground_m) \, p(t_n)
$$

where:

- $p(\label_{nm}\mid \ground_m,t_n)\
\begin{cases}
= t_n & \text{ iff } \label_{nm} = \ground_m\\
\propto 1-t_n & \text{ otherwise}\end{cases}$
- $p(\ground_m) \propto 1$
- $p(t_n) \propto 1$

The expectation of the complete log-likelihood then becomes:

$$
\newcommand{\obj}{\mathcal{Q}}
\newcommand{\param}{\mathbf{\theta}}
\newcommand{\expectation}{\mathbb{E}}
\begin{align}
\obj(\param,\param^{old})
&= \expectation_{\param^{old}} \ln p(\Ground,\Labels,\Trust)\\
&= \sum_n \sum_m \sum_{\ground_m} p(\ground_m | \Labels, \Trust, \param^{old}) \left( \ln p(\label_{nm}|\ground_m,t_n) + \ln p(\ground_m) + \ln p(t_n) \right)\\
&=  \sum_n \sum_m \sum_k \gamma_m(k) \ln p(\label_{nm}|\ground_m=k,t_n)\\
&=  \sum_n \sum_m \sum_k \gamma_m(k) I[\label_{nm}=k]\ln t_n + I[\label_{nm}\neq k] \ln (1-t_n)/c_m\\
\end{align}
$$
where $c_m$ is the cardinality of question $m$ minus one

Taking the derivative of the complete log-likelihood with respect to $t_n$ and setting it equal to zero, we get:
$\newcommand{\resp}{\gamma_m(k)}$
$$
\begin{align}
0 &= \sum_n \sum_{m\in\Labels_n} \sum_k \resp (I[l_{nm}=k] \, \frac{1}{t_n} - I[l_{nm}\neq k] \, \frac{c_m}{1-tn})\\
0 &= \sum_n \sum_{m\in\Labels_n} \sum_k \resp (I[l_{nm}=k] \, \frac{1-t_n}{t_n \, (1-t_n)} - I[l_{nm}\neq k] \, \frac{c_m \, t_n}{t_n \,(1-tn)})\\
0 &= \sum_n \sum_{m\in\Labels_n} \sum_k \resp (I[l_{nm}=k] \, (1-t_n) - I[l_{nm}\neq k] \, c_m \, t_n)
\end{align}
$$

where $\gamma_{m}(k) \triangleq p(\ground_m=k|\Labels_{.m},\Trust)$ can be computed using Bayes' theorem (E-Step):

$$
\newcommand{\annot}{\mathcal{A}}
\begin{align}
\gamma_{m}(k) &= \frac{\prod_{n \in \annot_m} p(\label_{nm}|\ground_m=k,t_n) \, p(g_m=k) \, p(t_n)}{ \sum_l \prod_{n \in \annot_m} p(\label_{nm}|\ground_m=l,t_n)\,p(\ground_m=l)\,p(t_n) }
\end{align}
$$

So in the M-step, we optimise $t_n$ as

$$
t_n = \frac{\sum_{m\in\Labels_n} I[\label_{nm}==\hat l]\gamma_{m}(k)}{\sum_{m\in\Labels_n} \gamma_{m}(k)}
$$
"""


"""
some references
em:
https://github.com/RafaeNoor/Expectation-Maximization/blob/master/EM_Clustering.ipynb
https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
https://www.jstor.org/stable/pdf/2346806.pdf?refreqid=excelsior%3A02dbe84713a99816418f5ddd3b41a93c&ab_segments=&origin=&acceptTC=1

annotator reliability:
https://dl.acm.org/doi/pdf/10.1145/1743384.1743478
https://aclanthology.org/D08-1027.pdf
https://aclanthology.org/P99-1032.pdf
https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.0006-341X.2004.00187.x

https://www.aaai.org/ocs/index.php/WS/AAAIW12/paper/view/5350/5599

OG file:
https://colab.research.google.com/drive/186F0yeqV_5OpIC4FkjHysEJ0tAiEW6Y-?authuser=1#scrollTo=XTLv7eS2NORn
"""



import numpy as np
import random
import pandas, pickle


def to_cat(y, num_classes=None, dtype=float):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def pickUniform(valueList):
    i = np.random.randint(0, len(valueList))
    return valueList[i]


class Question:
    def __init__(self, possibleValues=[True, False]):
        self.options = possibleValues

    def __repr__(self):
        return "Question(%s)" % (self.options)


class SimulatedQuestion(Question):
    """
    Sample a ground truth answer to a question
    """

    def __init__(self, possibleValues=[True, False], gt=None):
        self.options = possibleValues
        if gt:
            self.val = gt
        else:
            self.val = pickUniform(possibleValues)

    def __repr__(self):
        return "Question(%s) -> %s" % (self.options, self.val)


class Annotator:
    """
    Store the set of answers given by an annotator
    """

    def __init__(self, questionIndices, answers):
        #        self.idx = questionIndices   # store the the indices of the questions this annotator has answered
        #        self.ans = answers           # store the annotator's answers
        self.ans = {}
        for q, a in zip(questionIndices, answers):
            self.ans[q] = a
        self.t = .5 + .1 * r.rand()  # Initialise the annotators trustworthiness t
        self.logT = np.log(self.t)

    def getAnnotIndices(self):
        return list(self.ans.keys())

    def getAnnotAnswers(self):
        return [self.ans[k] for k in self.ans.keys()]

    def logprob(self, m, gt, numOptions):
        """
        Return the probability that this annotator would have given the answer
        they gave, given that gt is the real answer (and this annotator's
        trustworthiness
        """
        assert (m in self.ans)
        if self.ans[m] == gt:
            return self.logT
        else:
            return np.log((1 - self.t) / (numOptions - 1))  # spread
            # the probability mass over the *remaining* possible answers

    def __repr__(self):
        return "Annotator(%f) -> %s" % (self.t, self.ans)


class SimulatedAnnotator(Annotator):
    """
    Create an artificial annotator, by sampling this annotator's trustworthiness
    and generating answers according to this trustworthiness"""

    def __init__(self, nQuest, questions):
        self.t = .5 + np.random.uniform() / 2  # Assume that trustworthiness ranges from .5 to 1
        self.ans = {}
        totalQuest = len(questions)
        for _ in range(nQuest):
            m = r.randint(totalQuest)
            if m not in self.ans:
                # give the right answer with probability self.t
                s = r.uniform()
                if s < self.t:
                    self.ans[m] = questions[m].val
                else:
                    # else, sample a wrong answer
                    a = r.randint(len(questions[m].options))
                    while questions[m].options[a] == questions[m].val:
                        a = r.randint(len(questions[m].options))
                    self.ans[m] = questions[m].options[a]


class E():
    """
    p(l_nm | GT_m == k, t_n) is implemented as (t_n if k==l_nm else 1-t_n)
    p(g_m == k) and p(GT_m == l) implemented as (1/cardinality)
    """
    def __init__(self, K):
        self.N = np.arange(0,user.__len__()) # annotators
        self.M = np.arange(0,nQuestions) # questions
        self.L = np.arange(0,K) # given label per question
        self.K = K
        self.cm = K-1 # -1 because there's one good answer and the rest is wrong
        self.gamma_ = pandas.DataFrame(columns=[i for i in range(K)])

    def gamma(self, m, k):
        # n_Am_k = to_cat(user.loc[:,[f'q_{i}' for i in range(k)]]) # select question answered matrix and make categorical tensor from user table
        # n_am_k_inv = 1- n_Am_k
        #
        # tn_k = np.tile(user.loc[:,"t_model"],k)
        # tn_k_inv = (1-tn_k)/self.cm
        #
        # term1 = np.multiply(n_Am_k, tn_k)
        # term2 = np.multiply(n_am_k_inv, tn_k_inv)
        #
        # factor1 = np.add(term1,term2) # n x m x k
        # (factor1 / self.cm * (1 / self.K))

        # denom = sum(np.prod((factor1 / self.cm * (1 / self.K)), axis=0).__array__(), axis=1) # vector of k?

        # num =


        num = np.prod([
                        (n[1]["T_model"] if k == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm)) * (1 / self.K) * n[1]["T_model"]
                       for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ])
        denom = sum([
                    np.prod([
                            (n[1]["T_model"] if l == n[1][f"q_{m}"] else ((1 - n[1]["T_model"])/ self.cm )) * (1 / self.K) * n[1]["T_model"]
                              for n in user.loc[~np.isnan(user.loc[:,f"q_{m}"])].iterrows()
                    ]
                    ) for l in self.L])
        return num/denom

    # def Q(self):
    #     return sum([
    #                 sum([
    #                     sum(
    #                         [self.gamma(m,k)*(1 if label[m][k]==k else 0) * np.log(annotator.t[n]) +
    #                          (1 if label[m][k]!=k else 0) * np.log(1-annotator.t[n])/max(self.K) for k in self.K]
    #                         ) for m in self.M]
    #                     ) for n in self.N]
    #                 )

    def step(self):
        for m in self.M:
            for k in range(self.K):
                self.gamma_.loc[m,k] = self.gamma(m, k)
        return self.gamma_

class M():
    def __init__(self):
        pass

    def step(self, gamma, n):

        nom = sum([
                sum([
                    (gamma.loc[m,k] if k == n[f"q_{m}"] else 0)
                for k in list(range(car))])
              for m in list(range(nQuestions))])

        denom = sum([sum([gamma.loc[m,k] for k in range(car)])
            for m in list(range(nQuestions))])

        return nom/denom

if __name__ == "__main__":

    iterations = 5
    use_real = 0
    if use_real:

        with open('data/users.csv', 'r') as file:
            user = pandas.read_csv(file)

        with open('data/web_annotations.csv', 'r') as file:
            annotation = pandas.read_csv(file, names=["id", "object_label", "exclude", "rank", "anno_1_con_move",
                                                        "anno_1_uncon_move", "anno_1_dir_affs", "anno_1_indir_affs",
                                                        "anno_1_observe_affs", "anno_1_social_affs", "anno_1_no_affs",
                                                        "anno_1_no_clue", "anno_1_roll", "anno_1_push", "anno_1_drag",
                                                        "anno_1_tether", "anno_1_pick_up_carry", "anno_1_pour",
                                                        "anno_1_fragile", "anno_1_open", "anno_1_grasp", "anno_1_pull",
                                                        "anno_1_tip", "anno_1_stack", "anno_1_cut_scoop", "anno_1_support",
                                                      "anno_1_transfer", "anno_1_requires_other", "anno_1_info", "anno_1_deco",
                                                      "anno_1_together", "anno_1_none", "anno_1_warmth", "anno_1_illumination",
                                                      "anno_1_walk", "anno_1_id", "anno_2_con_move", "anno_2_uncon_move",
                                                      "anno_2_dir_affs", "anno_2_indir_affs", "anno_2_observe_affs",
                                                      "anno_2_social_affs", "anno_2_no_affs", "anno_2_no_clue", "anno_2_roll",
                                                      "anno_2_push", "anno_2_drag", "anno_2_tether", "anno_2_pick_up_carry",
                                                      "anno_2_pour", "anno_2_fragile", "anno_2_open", "anno_2_grasp",
                                                      "anno_2_pull", "anno_2_tip", "anno_2_stack", "anno_2_cut_scoop",
                                                      "anno_2_support", "anno_2_transfer", "anno_2_requires_other",
                                                      "anno_2_info", "anno_2_deco", "anno_2_together", "anno_2_none",
                                                      "anno_2_warmth", "anno_2_illumination", "anno_2_walk", "anno_2_id",
                                                      "anno_3_con_move", "anno_3_uncon_move", "anno_3_dir_affs",
                                                      "anno_3_indir_affs", "anno_3_observe_affs", "anno_3_social_affs",
                                                      "anno_3_no_affs", "anno_3_no_clue", "anno_3_roll", "anno_3_push",
                                                      "anno_3_drag", "anno_3_tether", "anno_3_pick_up_carry", "anno_3_pour",
                                                      "anno_3_fragile", "anno_3_open", "anno_3_grasp", "anno_3_pull",
                                                      "anno_3_tip", "anno_3_stack", "anno_3_cut_scoop", "anno_3_support",
                                                      "anno_3_transfer", "anno_3_requires_other", "anno_3_info", "anno_3_deco",
                                                      "anno_3_together", "anno_3_none", "anno_3_warmth", "anno_3_illumination",
                                                      "anno_3_walk", "anno_3_id"])
    else:
        with open('simulation data/uniform_user.pickle', 'rb') as file:
            user = pickle.load(file)
        with open('simulation data/uniform_annotations_empty.pickle', 'rb') as file:
            annotations = pickle.load(file)

    # init user weights at 1
    for i in range(iterations+1):
        user[f'q_weight_{i}'] = np.ones(user.__len__()) * 0.5 # all users start at weight 0.5 as prob(good|agree) is 0.5 at starting time
    user['included'] = np.ones(user.__len__())


    nAnnot = user.__len__()
    nQuestions = annotations.__len__()
    # x = np.linspace(0,1,10**3)
    car = 5
    #
    # udata = {"ID":range(nAnnot),
    #          "T_given": random.choices(x,k=nAnnot),
    #          "T_model": np.zeros(nAnnot)}
    #
    # for q in range(nQuestions): # keep track of labels in broad format
    #     udata[f"q_{q}"] = np.zeros(nAnnot)
    #
    # user = pandas.DataFrame(data=udata)



    # # annotator.q_answered == questions answered by this annotator
    # question = pandas.DataFrame(data={"GT":random.choices(range(car), k=nQuestions),
    #                                   "model": np.zeros(nQuestions),
    #                                   "id_1": np.zeros(nQuestions),
    #                                   "annot_1": np.zeros(nQuestions),
    #                                   "id_2": np.zeros(nQuestions),
    #                                   "annot_2": np.zeros(nQuestions),
    #                                   "id_3": np.zeros(nQuestions),
    #                                   "annot_3": np.zeros(nQuestions)
    #                                   })
    e = E(car)
    m = M()
    i = 0
    while i < iterations:

        g = e.step()
        for ann in user.ID:
            user.loc[ann, "T_model_next"] = m.step(g, user.iloc[ann])
        user.loc[:, "T_model"] =user.loc[:, "T_model_next"]
        i += 1

