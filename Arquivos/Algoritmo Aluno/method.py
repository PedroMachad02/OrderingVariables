from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import LabelEncoder
from pgmpy.estimators import BDeuScore, K2Score
from pgmpy.readwrite import BIFWriter
import networkx as nx
import pandas as pd
import numpy as np
import time
import random
import sys


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


def spanning_tree_maximization(G, SET, D):
    nodes_to_maximize = [v for v in G.nodes]
    
    bdeusc = BDeuScore(D)
    scores = {v: bdeusc.local_score(v, SET[v]) for v in G.nodes}
    
    #o algorítmo de arvore geradora implementada no networkx retorna
    #árvores geradoras para cada componente, isso faz com que ele funcione para grafos desconexos
    #árvores geradoras = florestas abrangentes
    
    while nodes_to_maximize != [] and list(G.edges) != []:
        sp_tree = nx.maximum_spanning_tree(G, algorithm='kruskal')
        SET_theta = {v: [] for v in G.nodes} #pais candidatos
        for x, y in sp_tree.edges:
            SET_theta[x].append(y)
            SET_theta[y].append(x)
            G.remove_edge(x, y)

        for x in nodes_to_maximize:
            curr_sc = scores[x]
            new_sc = curr_sc
            maximizing = True
            while maximizing:
                new_pa = None
                for y in SET_theta[x]:
                    sc = bdeusc.local_score(x, SET[x] + [y])
                    if sc > new_sc:
                        new_sc = sc
                        new_pa = y
                if new_sc != curr_sc:
                    SET[x].append(new_pa)
                    SET_theta[x] = [y for y in SET_theta[x] if y != new_pa]
                    curr_sc = new_sc
                else:
                    if curr_sc != scores[x]:
                        scores[x] = curr_sc
                    else:
                        nodes_to_maximize = [v for v in nodes_to_maximize if v != x]
                    maximizing = False


def psi(N):
    return np.log2(N) / 2

def cpt(x, pa, V_CARD):
    card = V_CARD[x] - 1
    if len(pa) == 0:
        return V_CARD[x]
    for p in pa:
        card *= V_CARD[p]
    return card

def pen(x, pa, N, V_CARD):
    return -psi(N) * cpt(x, pa, V_CARD)


def mcond_entropy(x, y, D):
    if len(y) == 0:
        return drv.entropy(D[x], base=2)
    x = D[x]
    y = [D[v] for v in y]
    yt = tuple([tuple(r) for r in y])
    xyt = tuple([tuple(r) for r in [x] + y])
    return drv.entropy_joint(np.array(xyt), base=2) - drv.entropy_joint(np.array(yt), base=2)

def ig(x, pa, D):
    return float(drv.entropy(D[x], base=2)) - float(mcond_entropy(x, pa, D))


def variable_ordering(PI):
    #ordenação            
    variables = [v for v in V]
    order = []
    while variables != []:
        var = None
        min_gain = float('inf')
        for x in variables:
            pa = [y for y in PI[x] if y in order]
            pa2 = [y for y in PI[x]]
            gain = (mcond_entropy(x, pa, D) - mcond_entropy(x, PI[x], D)) * (cpt(x, pa, V_CARD) / cpt(x, PI[x], V_CARD))
            if gain < min_gain:
                min_gain = gain
                var = x
        order.append(var)
        variables = [v for v in variables if v != var]
    return order


    variables = V
    ordering = []
    while variables != []:
        var = None
        min_gain = float('inf')
        for x in variables:
            pa = [y for y in PI[x] if y not in ordering]
            gain = ig(x, pa, D) * (pen(x, PI[x], N, V_CARD) - pen(x, pa, N, V_CARD))
            if gain < min_gain:
                min_gain = gain
                var = x
        ordering.append(var)
        variables = [v for v in variables if v != var]
    return ordering


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
    number = ''
    if len(sys.argv) == 5:
        number = '_' + sys.argv[4]


    DATA_CSV = sys.argv[1]

    D, V, N, V_CARD = load_data(DATA_CSV)
    V = list(V)
    random.shuffle(V)
    print('Num vars:', len(V))
    print('Num samples:', N)

    #Método
    start_time = time.time()
    I, comb = all_mutual_information(D, V)
    print('Num combinations:', len(comb))

    G = nx.Graph(
        [(x, y, {'weight': I[x][y]}) for x, y in comb]
    )

    # 1ª parte: maximização com AGM  
    PI_theta = {v: [] for v in V}
    spanning_tree_maximization(G, PI_theta, D)

    # 2ª parte: mantendo concorrências
    for x in PI_theta:
        to_remove = []
        for y in PI_theta[x]:
            if x not in PI_theta[y]:
                to_remove.append(y)
        PI_theta[x] = [y for y in PI_theta[x] if y not in to_remove]

    # salvando arestas concorrentes no txt
    if sys.argv[2] == 'save_concur':
        snd_stp_edg = []
        for x in PI_theta:
            for y in PI_theta[x]:
                if (y, x) not in snd_stp_edg:
                    snd_stp_edg.append((x, y))

        with open('results/' + DATA_CSV.split('/')[1].split('.')[0] + '_concurrences' + number + '.txt', 'w') as f:
            f.write(str(time.time() - start_time) + '\n')
            for y, x in snd_stp_edg:
                f.write(y + ',' + x + '\n')

    # 3ª parte: ordenação de variáveis
    ordering = None
    _sorted_ = ''
    if sys.argv[3] == 'sorted':
        ordering = variable_ordering(PI_theta)
        _sorted_ = '_sorted'
    else:
        _sorted_ = '_random'
        ordering = V

    # 4ª parte: número máximo de pais
    max_parents = {}
    for v in V:
        max_parents[v] = len(PI_theta[v])
        if len(PI_theta[v]) == 0:
            max_parents[v] = 1

    # 5ª parte: algoritmo K2
    PI = K2_algorithm(ordering, max_parents, D)

    final_time = time.time() - start_time

    method_edges = []
    for x in PI:
        for y in PI[x]:
            method_edges.append((y, x))

    with open('results/' + DATA_CSV.split('/')[1].split('.')[0] + '_method' + _sorted_ + number + '.txt', 'w') as f:
        f.write(str(final_time) + '\n')
        for y, x in method_edges:
            f.write(y + ',' + x + '\n')