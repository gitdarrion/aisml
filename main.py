import numpy as np

import random
from collections import deque

def affinity(ag,Ab):
    
    result = np.array([np.linalg.norm(ag-ab) for ab in Ab])
    return result 

def select(f, Ab, n):
    fAb = sorted([(f,Ab) for f,Ab in zip(f, Ab)], key=lambda k:k[0], reverse=True)
    Abn = [ab[1] for ab in fAb]
    return Abn[:n]

def clone(Abn, B, f):
    clones = []
    for i in range(0,len(Abn)):
        Nc = round((B*len(Abn))/(i+1))
        clones += ([Abn[i]]*Nc)
    return clones

def hypermutate(C, f):

    for i in range(len(C)):
        shift = random.randint(0,len(C[0]))
        C[i] = np.roll(C[i], shift)
    return C

def insert(ag, ab_alpha, ab_beta): 
    a,b = [affinity(ag, ab_alpha), affinity(ag, ab_beta)]
    return max(a,b)

def clonalg(Ab, Ag, gen, n, d, L, B, M, N):
    f = []
    Abm = []
    C = []

    for i in range(0,gen): 
        for j in range(0,M):
            f.append(affinity(Ag[j], Ab))
            Abn = select(f[j], Ab, n)
            C += clone(Abn, B, f[j])
            f_prime = affinity(Ag[j], C)
            C_prime = hypermutate(C, f_prime)
            f_prime = affinity(Ag[j], C_prime)
            ab = select(f_prime, C_prime, 1)
            Abm.append(ab if (len(Abm) < j+1) else insert(Ag[j], ab, Abm[j]))
        print(i)
M=50
N=10
n=5
d=1
L=5
B=1
gen=5
clonalg(
    Ab=np.random.randint(0,2,(N,L)),
    Ag=np.random.randint(0,2,(M,L)),
    gen=gen,
    n=n,
    d=d,
    L=L,
    B=B,
    M=M,
    N=N
)