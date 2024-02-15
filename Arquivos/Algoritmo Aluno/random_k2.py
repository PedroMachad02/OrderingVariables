import pandas as pd
import sys
from pgmpy.estimators import K2Score
from sklearn.preprocessing import LabelEncoder
import time
import random

def load_data(DATA_CSV):
    D = pd.read_csv(DATA_CSV, sep=',')
    D = D.apply(LabelEncoder().fit_transform)
    
    V = D.columns
    N = len(D.index)
    V_CARD = {v: len(D[v].unique()) for v in V}
    
    return D, V, N, V_CARD


def all_mutual_information(D, V):
    comb = []
    for x in V:
        for y in V:
            if x != y:
                t = (x, y)
                if t not in comb and t[::-1] not in comb:
                    comb.append(t)

    I = {v: {} for v in V}
    for x, y in comb:
        i = drv.information_mutual(D[x], D[y], base=2)
        I[x][y] = i
        I[y][x] = i
    
    return I, comb

def predecessors(x, pa, ordering):
    pred = []
    x_idx = ordering.index(x)
    for y in ordering[:x_idx]:
        if y not in pa:
            pred.append(y)
    return pred

def K2_algorithm(ordering, max_parents, D):
    k2sc = K2Score(D).local_score
    PI = {v: [] for v in V}
    for x in V[1:]:
        proceed = True
        old_sc = k2sc(x, PI[x])
        while proceed and len(PI[x]) < max_parents[x]:
            proceed = False
            z, new_sc = None, None
            pred = predecessors(x, PI[x], ordering)
            if pred != []:
                z, new_sc = max([(y, k2sc(x, PI[x] + [y])) for y in pred], key=lambda t: t[1])
                if new_sc > old_sc:
                    old_sc = new_sc
                    PI[x].append(z)
                    proceed = True
    return PI


if __name__ == '__main__':
    DATA_CSV = sys.argv[1]
    k = sys.argv[2]

    D, V, N, V_CARD = load_data(DATA_CSV)
    V = list(V)
    print('Num vars:', len(V))
    print('Num samples:', N)

    start_time = time.time()
    random.shuffle(V)
    
    parents = int(len(V)) / 5
    max_parents = {v: parents for v in V}
    ordering = [v for v in V]

    PI = K2_algorithm(ordering, max_parents, D)

    final_time = time.time() - start_time

    method_edges = []
    for x in PI:
        for y in PI[x]:
            method_edges.append((y, x))

    with open('results/' + DATA_CSV.split('/')[1].split('.')[0] + '_random_' + k + '.txt', 'w') as f:
        f.write(str(final_time) + '\n')
        for y, x in method_edges:
            f.write(y + ',' + x + '\n')