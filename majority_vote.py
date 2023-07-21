import numpy as np

class majority():
    def __init__(self, annotations, nQuestions, car, dup, users):
        self.maj_ans =[]
        self.disc_annot = set([]) # discarded annotators
        self.dup = dup
        self.users = users
        self.car = car
        self.set_pc(annotations, nQuestions)

    def set_discard(self, annotations):

        for q in annotations.loc[annotations['KG']==1, :].iterrows():
            for d in range(self.dup):
                if q[1][f'annot_{d}'] != q[1]['GT']:
                    self.disc_annot.add(q[1][f'id_{d}'])
        for u in self.users.loc[self.users['type'] == 'KG', :].iterrows():
            for d in range(self.dup):
                for q in annotations.loc[annotations[f'id_{d}'] == u[0], :].iterrows():# for all questions answered by a KG annotator
                    for d_ in range(self.dup): # go over all the other annotations for this question
                        if q[1][f'annot_{d_}'] != u[1][f'q_{q[0]}']:
                            self.disc_annot.add(q[1][f'id_{d_}'])
        if self.disc_annot.__len__() > 0:
            print(f'annotators {self.disc_annot} were removed')
        if self.disc_annot.__len__() == self.users.__len__():
            print(f'No annotators left who agree with all known good questions/annotators')
            return [np.nan]*annotations.__len__() # don't bother figuring out pc if there are no annotators left

    def set_pc(self, annotations, nQuestions):
        self.pc = np.sum(annotations['GT'] == self.run(annotations, nQuestions, with_disc=False))/ nQuestions
        if np.any(annotations['KG'])| np.any(self.users['type'] == 'KG'):
            self.pc_KG = np.sum(annotations['GT'] == self.run(annotations,nQuestions, with_disc=True))/ nQuestions
        else:
            self.pc_KG = self.pc

    def run(self, annotations,nQuestions, with_disc = False):
        self.maj_ans = []
        if with_disc:
            self.set_discard(annotations)
        else:
            self.disc_annot = set([])

        for q in range(nQuestions):
            # weights for all k options list
            if annotations.at[q, f'KG']:
                self.maj_ans.append(annotations.at[q, f'GT'])
            else:
                k_weight = np.zeros(self.car)
                for k in range(self.car):
                    # counter for number of people who chose option k
                    for d in range(self.dup):
                        if not annotations.at[q, f'id_{d}'] in self.disc_annot:
                            if annotations.at[q, f'annot_{d}'] == k:
                                k_weight[k] +=1


                if not np.array_equal(np.zeros(self.car),k_weight):
                    max_val = max(k_weight)
                    max_indices = []
                    for i, k in enumerate(k_weight):
                        if k == max_val:
                            max_indices.append(i)
                    self.maj_ans.append(max_indices[np.random.randint(max_indices.__len__())])
                else:
                    self.maj_ans.append(np.nan)
        return self.maj_ans