import pickle, pandas
import numpy as np


import itertools as it



class inspect():
    def __init__(self, mode, dup, car, p_fo):
        self.p_fo = p_fo
        self.dup = dup
        self.car = car
        with open(f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_user.pickle',
                  'rb') as file:
            self.user = pickle.load(file)
        with open(
                f'simulation data/{mode}_dup-{dup}_car-{car}_p-fo-{p_fo}_annotations_empty.pickle',
                'rb') as file:
            self.ann = pickle.load(file)

    def __call__(self, *args, **kwargs):
        sum_fo = sum([self.count_fo_correct(nid) for nid in range(self.dup)])
        print(f'sum_fo {sum_fo} for total {self.dup * self.ann.__len__()} options')
        sum_fo= sum_fo/(self.dup * self.ann.__len__())

        pcs_1 = {}
        pcs_2 = {}
        for gt in gts:
            print(f"\ngt: {gt} dup: {self.dup} p_fo: {self.p_fo} car: {self.car}")
            pcs_1[f'p_gt_{gt}']= self.count_more_than_one_FO(gt, 1) / (self.ann.loc[(self.ann["GT"] == gt)].count()[1]) * 100
            pcs_2[f'p_gt_{gt}']= self.count_more_than_one_FO(gt, 2) / (self.ann.loc[(self.ann["GT"] == gt)].count()[1]) * 100
            for cnt in cnts:
                print(f'{self.count_more_than_one_FO(gt, cnt):.1f} times >={cnt} FO annotators with n(GT) '
                      f'{self.ann.loc[(self.ann["GT"] == gt)].count()[1]}: '
                      f'pencentage: {self.count_more_than_one_FO(gt, cnt) / (self.ann.loc[(self.ann["GT"] == gt)].count()[1]) * 100:.1f}')
        mean1 = np.mean([pcs_1[key] for key in pcs_1.keys()])
        mean2 = np.mean([pcs_2[key] for key in pcs_2.keys()])

        return mean1, mean2, sum_fo, self

    def count_fo_correct(self, id):
        return sum((self.ann['GT'] == 0) & (self.user.loc[self.ann[f'id_{id}'], 'type'] == 'first_only'))


    def fos(self, ids):
        # return boolean array of first only encounters on given ids
        # folist is nquestions x len(ids)
        folist =  [(self.user.loc[self.ann[f'id_{id}'], 'type'] == 'first_only').values for id in ids]
        # if all ids have type first only for a given question, return true. Returns list of size nquestions
        return np.all(folist, axis=0)


    def count_more_than_one_FO(self, GT_val, n):

        combs = list(it.combinations(np.unique(np.arange(dup)), n))
        counts = [self.ann.loc[(self.ann['GT'] == GT_val) & self.fos(comb), 'GT'].count() for comb in combs]
        return sum(counts)/len(combs)


if __name__ == "__main__":
    # dup = 7
    # p_fo = 0.3
    # car = 5
    # mode = 'uniform'

    # dup = 7
    # p_fo = 0.3
    # car = 5
    # mode = 'uniform'

    # how many fo users to detect (max=dup)
    # cnts = range(1, 5)
    # gts = range(car)
    inspection_data = pandas.DataFrame(columns=['dup', 'p_fo', 'car', 'mode', 'per_fo_1', 'per_fo_2', 'fo_correct', 'obj'])

    T_dist = 'uniform'
    cars = list(range(3, 8))
    dups = [3, 5, 7, 9]
    p_fos = [0.0, 0.1, 0.2, 0.3]

    # cars = list(range(7, 8))
    # dups = [9]
    # p_fos = [0.3]

    # cars = [2]
    # dups = [3]
    # p_fos = [.3]
    for dup in dups:
        for p_fo in p_fos:
            for car in cars:
                gts = range(car)
                cnts = range(1,4)
                inspection_data.loc[inspection_data.__len__(), :] = [dup, p_fo, car, T_dist, *inspect(T_dist, dup, car, p_fo).__call__(gts=gts, cnts=cnts)]
    pass

# inspection_data.loc[inspection_data['p_fo']==0.3]
