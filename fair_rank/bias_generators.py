import numpy as np
from itertools import combinations, permutations
from time import time
import pandas as pd
from numpy.random import random
import warnings
warnings.filterwarnings('ignore')
from scipy.special import comb
from scipy import stats
from scipy.spatial import distance
from agg_util import *
from fare.metrics import *
import seaborn as sns
from ir_metrics import *
import fairsearchcore as fsc
from yang_metrics import *
from dataGenerator import generateUnfairRanking


def gen_minority_group_idx(n, p, ratio):
    len0 = int(n*ratio)
    len1= n - len0
    idxs = []
    i=0
    while len1>0 and len0>0:
        ran = random() 
        if ran < p:
            idxs.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        idxs.append(i)
        len0 -= 1
        i+=1
    return idxs


def bias_desending_idx(n, p, ratio,d):
    len0 = int(n*ratio)
    len1 = n - len0
    idxs = []
    i=0
    while len1>0 and len0>0:
        if random() < p: 
            idxs.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
#         print(p)
        p -= d
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        idxs.append(i)
        len0 -= 1
        i += 1
    return idxs

def bias_desending_rooney(n, p, ratio, d, k, rooney):
    len0 = int(n*ratio)
    len1 = n - len0
    idxs = []
    i=0
    while len1>0 and len0>0:
        if random() < p: 
            idxs.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
#         print(p)
        p -= d
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        idxs.append(i)
        len0 -= 1
        i += 1
        
    min_prot_top = rooney*k
    num_prot_top = sum(1 for i in idxs if i < k)

    if num_prot_top < min_prot_top:
        #swap extra protected to satisfy rooney requirement
        need = min_prot_top - num_prot_top
        bottom = [x for x in idxs if x >=k]
        bottom = np.random.choice(bottom, need, replace=False)
        idxs = [x for x in idxs if x not in bottom]
        top = [x for x in range(k) if x not in idxs]
        to_add = np.random.choice(top, need, replace=False)
        idxs = np.concatenate([idxs, to_add])
    return idxs


# top spots in the ranking are padded to 
#start with k candidates from non-protected group
#then rest of the ranking has normal bias given by p
def padded_idx(n, p, ratio, k):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len1:
        raise TypeError("need at least k candidates from non-protected group")
    idxs = []
    i=k #first k spots given to non-protected
    len1 = len1-k
    while len1>0 and len0>0:
        if random() < p: 
            idxs.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        idxs.append(i)
        len0 -= 1
        i += 1
    return idxs


# top spots in the ranking are padded to 
#start with k candidates from protected group
#then rest of the ranking has normal bias given by p
def padded_protect_biased(n, p, ratio, k):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len0:
        raise TypeError("need at least k candidates from protected group")
    idxs = np.arange(k) #first k spots given to protected
    sffx = []
    len0 = len0-k
    i=k
    while len1>0 and len0>0:
        if random() < p: 
            sffx.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        sffx.append(i)
        len0 -= 1
        i += 1
    return np.concatenate((idxs,sffx))


# top spots in the ranking are padded to 
#start with k candidates from non-protected group
#then rest of the ranking has normal bias given by p
def padded_protect(n, ratio, k):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len0:
        raise TypeError("need at least k candidates from protected group")
    idxs = np.arange(k) #first k spots given to protected
    sffx = np.random.choice(np.arange(k,n),len0-k, replace=False)
    return np.concatenate((idxs,sffx))


# top spots in the ranking are padded to 
#start with k candidates from non-protected group
#plus 10% token candidate from the protected
#then rest of the ranking has normal bias given by p
def padded_random(n, ratio, k):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len1:
        raise TypeError("need at least k candidates from non-protected group")
    #first k spots given to non-protected
    return np.random.choice(np.arange(k,n), len0, replace=False)


# top spots in the ranking are padded to 
#start with k candidates from non-protected group
#plus one token candidate from the protected
#then rest of the ranking has normal bias given by p
def padded_rooney_random(n, ratio, k, c):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len1:
        raise TypeError("need at least k candidates from non-protected group")
    if c > len0:
        raise TypeError("need at least c candidates from protected group")
    # first k spots given to non-protected
    # choose c protected group candidate to include in top-k
    idxs = np.random.choice(np.arange(k), c, replace=False)
    sffx = np.random.choice(np.arange(k,n),len0-c, replace=False)
    idxs= np.concatenate((idxs,sffx))
    return idxs

# top spots in the ranking are padded to 
#start with k candidates from non-protected group
#plus 10% token candidate from the protected
#then rest of the ranking has decreasing bias given by p
def padded_rooney_decrease(n, ratio, k, p, c):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len1:
        raise TypeError("need at least k candidates from non-protected group")
    if c > len0:
        raise TypeError("need at least c candidates from protected group")
    # first k spots given to non-protected
    # choose c protected group candidate to include in top-k
    idxs = np.random.choice(np.arange(k), c, replace=False)
    sffx = []
    i=k #first k spots given to non-protected
    len1 = len1-k+c
    len0 = len0-c
    while len1>0 and len0>0:
        if random() < p: 
            sffx.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        sffx.append(i)
        len0 -= 1
        i += 1
    idxs= np.concatenate((idxs,sffx))
    return idxs


################################################################################

# fixed bias
def fixed_bias(n_ranks, rank_len, ps, rs):
    fixed = {}
    for r in rs:
        print(r)
        fixed[r] ={}
        for p in ps:
            fixed[r][p] =[]
            for i in range(n_ranks):
                g = gen_minority_group_idx(rank_len, p, r)
                fixed[r][p].append(g)
    return fixed


# decreasing bias
def decreasing_bias(n_ranks, rank_len, ps, rs,d):
    decrease = {}
    for r in rs:
        print(r)
        decrease[r] ={}
        for p in ps:
            decrease[r][p] =[]
            for i in range(n_ranks):
                g = bias_desending_idx(rank_len, p, r, d)
                decrease[r][p].append(g)
    return decrease

# decreasing bias
def decreasing_bias_rooney(n_ranks, rank_len, ps, rs, d, k, roo):
    decrease = {}
    for r in rs:
        print(r)
        decrease[r] ={}
        for p in ps:
            decrease[r][p] =[]
            for i in range(n_ranks):
                g = bias_desending_rooney(rank_len, p, r, d, k, roo)
                decrease[r][p].append(g)
    return decrease


# padded top-k random bottom n-k
def pad_random(n_ranks, rank_len, ks, rs):
    random_pad = {}
    for r in rs:
        print(r)
        random_pad[r] ={}
        for k in ks:
            if rank_len*(1-r) >= k:
                random_pad[r][k]=[]
                for i in range(n_ranks):
                    g = padded_random(rank_len, r, k)
                    random_pad[r][k].append(g)
    return random_pad


def padded_idx(n, p, ratio, k):
    len0 = int(n*ratio)
    len1 = n - len0
    if k > len1:
        raise TypeError("need at least k candidates from non-protected group")
    idxs = []
    i=k #first k spots given to non-protected
    len1 = len1-k
    while len1>0 and len0>0:
        if random() < p:
            idxs.append(i)
            len0 -= 1
        else:
            len1 -= 1
        i += 1
    while len1 > 0:
        len1 -= 1
    while len0 >0:
        idxs.append(i)
        len0 -= 1
        i += 1
    return idxs


# padded top-k p bias bottom n-k
def pad_biased(n_ranks, rank_len, ks, rs, p):
    pad = {}
    for r in rs:
        print(r)
        pad[r] ={}
        for k in ks:
            if rank_len*(1-r) >= k:
                pad[r][k]=[]
                for i in range(n_ranks):
                    g = padded_idx(rank_len, p, r, k)
                    pad[r][k].append(g)
    return pad


# padded top-k random bottom n-k
def pad_random_rooney(n_ranks, rank_len, ks, rs):
    rooney = {}
    for r in rs:
        print(r)
        rooney[r] ={}
        for k in ks:
            if rank_len*(1-r) >= k:
                c = np.max([1, int(k*0.1)])
                rooney[r][k]=[]
                for i in range(n_ranks):
                    g = padded_rooney_random(rank_len, r, k, c)
                    rooney[r][k].append(g)
    return rooney


# padded top-k random bottom n-k
def pad_random_rooney1(n_ranks, rank_len, ks, rs):
    rooney = {}
    for r in rs:
        print(r)
        rooney[r] ={}
        for k in ks:
            if rank_len*(1-r) >= k:
                rooney[r][k]=[]
                for i in range(n_ranks):
                    g = padded_rooney_random(rank_len, r, k, 1)
                    rooney[r][k].append(g)
    return rooney

# padded top-k random bottom n-k
def pad_decrease_rooney(n_ranks, rank_len, ks, rs, p):
    rooney = {}
    for r in rs:
        print(r)
        rooney[r] ={}
        for k in ks:
            if rank_len*(1-r) >= k:
                c = np.max([1, int(k*0.1)])
                rooney[r][k]=[]
                for i in range(n_ranks):
                    g = padded_rooney_decrease(rank_len, r, k, p, c)
                    rooney[r][k].append(g)
    return rooney


# padded top-k random bottom n-k
def pad_protected_random(n_ranks, rank_len, ks, rs):
    random_pad = {}
    for r in rs:
        print(r)
        s = rank_len*r
        random_pad[r] ={}
        for k in ks:
            if s >= k:
                random_pad[r][k]=[]
                for i in range(n_ranks):
                    g = padded_protect(rank_len, r, k)
                    random_pad[r][k].append(g)
    return random_pad


# padded top-k random bottom n-k
def pad_protected_bias(n_ranks, rank_len, ks, rs, p):
    random_pad = {}
    for r in rs:
        print(r)
        s = rank_len*r
        random_pad[r] ={}
        for k in ks:
            if s >= k:
                random_pad[r][k]=[]
                for i in range(n_ranks):
                    g = padded_protect_biased(rank_len, p, r, k)
                    random_pad[r][k].append(g)
    return random_pad

