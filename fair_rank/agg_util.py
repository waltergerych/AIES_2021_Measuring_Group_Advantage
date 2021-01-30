"""
This code was adapted from example:
https://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html


    References
    ----------
    V Conitzer, A Davenport, J Kalagnanam. 
    Improved bounds for computing Kemeny rankings, 2006
    
    Caitlin Kuhlman, MaryAnn VanValkenburg, Elke Rundensteiner. 
    "FARE: Diagnostics for Fair Ranking using Pairwise Error Metrics" 
    in the proceedings of the Web Conference (WWW 2019)
"""

# Authors: Caitlin Kuhlman <cakuhlman@wpi.edu>
# License: BSD 3 clause


import numpy as np
from numpy.random import random
from itertools import combinations, permutations
from time import time
#from gurobipy import *
from fare.metrics import _count_inversions,_merge_parity,_merge_eq

################################# Distances  ####################################
def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) ==
                -np.sign(rank_b[i] - rank_b[j]))
    return tau

#brute force solution for Kemeny aggregation
def rankaggr_brute(ranks):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return min_dist, best_rank

#build matrix C to represent rankings to be aggregated
def build_graph(ranks):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights


#compute rank parity score for a ranking
def rank_parity(y,groups):
    r = np.transpose([y,groups])
    r = r[r[:,0].argsort()]
    g= np.array(r[:,1], dtype=int)
    e0 = _count_inversions(g, 0, len(g)-1, _merge_parity, 0)[1]
    e1 = _count_inversions(g, 0, len(g)-1, _merge_parity, 1)[1]

    return e0,e1


#brute force a solution with parity constraint
def parity_brute(ranks, groups, thresh):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        e = rank_parity(candidate_rank, groups)
        if e <= thresh:
            dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
            if dist < min_dist:
                min_dist = dist
                best_rank = candidate_rank
    return min_dist, best_rank



#brute for ce aggregation with equality constraints
def equality_brute(ranks, groups, thresh):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        e0 = np.sum([rank_equality(candidate_rank, rank, groups)[0] for rank in ranks])
        e1 = np.sum([rank_equality(candidate_rank, rank, groups)[1] for rank in ranks])
        edist = abs(e0-e1)
        if edist <= thresh:
            kdist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
            if kdist < min_dist:
                min_dist = kdist
                best_rank = candidate_rank
    return min_dist, best_rank


#compute rank equality score for each group in a ranking
def rank_equality(y_true, y_pred, groups):
    #sort instances by y_pred
    r = np.transpose([y_true,y_pred,groups])
    r = r[r[:,0].argsort()]
    e0 = _count_inversions(r, 0, len(r)-1, _merge_eq, 0)[1]
    e1 =  _count_inversions(r, 0, len(r)-1, _merge_eq, 1)[1]
    return e0, e1

########################### AGGREGATION ###############################


#build matrix for parity constraints
def build_parity_constraints(groups):
    n_candidates = len(groups)
    edges = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        
        edges[i, j] = (groups[i] - groups[j])
        edges[j,i] = -(groups[i] - groups[j])
    
    return edges.ravel()



#build precedence matrix for rankings being aggregated
def all_pair_precedence(ranks, groups):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        e_ij = groups[i] - groups[j]   # preferred group 
        e_ji = groups[j] - groups[i]   # preferred group 
        edge_weights[i, j] = h_ij
        edge_weights[j, i] = h_ji
    for i, j in combinations(range(n_candidates), 2):
        e_ij = groups[i] - groups[j]   # preferred group 
        e_ji = groups[j] - groups[i]   # preferred group 
        edge_weights[i, j] *= e_ij
        edge_weights[j, i] *= e_ji
    return edge_weights.T.ravel()



# p = 0.5 -> random, fair
# p large -> unfair favors g1
def gen_groups(n, p):
    len1 = int(n/2)
    len0 = n - len1
    groups = []
    while len1 > 0 and len0 >0:
        if random() < p: 
            groups.append(1)
            len1 -= 1
        else:
            groups.append(0)
            len0 -= 1
    while len1 > 0:
        groups.append(1)
        len1 -= 1
    while len0 >0:
        groups.append(0)
        len0 -= 1
    return groups


def aggregate_kemeny(n_voters, n_candidates, ranks):

    #Declare gurobi model object
    m = Model()
    m.setParam("OutputFlag", 0);

    # Indicator variable for each pair
    x = {}
    c=0
    for i in range(n_candidates):
        for j in range(n_candidates):
            x[c] = m.addVar(vtype=GRB.BINARY, name="x(%d)(%d)" %(i,j))
            c+=1
    m.update()

    idx = lambda i, j: n_candidates * i + j

    # pairwise constraints
    for i, j in combinations(range(n_candidates), 2):
        m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)
    m.update()   

    #transitivity constraints
    for i, j, k in permutations(range(n_candidates), 3):
        m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k,i)] >= 1)
    m.update()

    # Set objective
    # maximize c.T * x
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()
    m.setObjective(quicksum(c[i]*x[i] for i in range(len(x))), GRB.MAXIMIZE)
    m.update()
    #m.write("kemeny_n"+str(n_ranks)+"_N"+str(rank_len) + "_t"+str(theta) + ".lp")
    t0 = time()
    m.optimize()
    t1 = time()

    if m.status == GRB.OPTIMAL:
        #m.write("kemeny_n"+str(rank_len)+"_N"+str(n_ranks)+"_t"+str(theta)+".sol")

        #get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol=np.array(sol)
        aggr_rank = np.sum(sol.reshape((n_candidates,n_candidates)), axis=1)
        return aggr_rank,t1-t0
    else:
        return None,t1-t0

def aggregate_sparse(n_voters, n_candidates, ranks, thresh):
           #Declare gurobi model object
        m = Model()
        m.setParam("OutputFlag", 0);

        # Indicator variable for each pair
        x = {}
        for i in range(n_candidates**2):
            x[i] = m.addVar(vtype=GRB.BINARY, name="x%d" % i)

        m.update()

        # pairwise constraints
        for i, j in combinations(range(n_candidates), 2):
            m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)

        m.update()   

        #transitivity constraints
        for i, j, k in permutations(range(n_candidates), 3):
            m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k,i)] >= 1)
        m.update()

        coefs =[]
        xs =[]
        for i, j in combinations(range(n_candidates), 2):
            preference = ranks[:, i] - ranks[:, j]
            h_ij = np.sum(preference < 0)  # prefers i to j
            h_ji = np.sum(preference > 0)  # prefers j to i
            if h_ij > h_ji:
                xs.append(x[idx(i,j)])
                coefs.append(-1*(h_ij - h_ji))
            elif h_ij < h_ji:
                xs.append(x[idx(j,i)])
                coefs.append(-1*(h_ji - h_ij)) 
        m.setObjective(LinExpr(coefs, xs), GRB.MAXIMIZE)
        m.update()
        # m.write("latest_model.lp")
        m.optimize()
        # m.write("latest_model.sol")    

        #get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol=np.array(sol)
        return np.sum(sol.reshape((n_candidates,n_candidates)), axis=1)

def aggregate_parity(n_voters, n_candidates, ranks, groups, thresh):

    #Declare gurobi model object
    m = Model()
    m.setParam("OutputFlag", 0);

    # Indicator variable for each pair
    x = {}
    c=0
    for i in range(n_candidates):
        for j in range(n_candidates):
            x[c] = m.addVar(vtype=GRB.BINARY, name="x(%d)(%d)" %(i,j))
            c+=1
    m.update()

    idx = lambda i, j: n_candidates * i + j

    # pairwise constraints
    for i, j in combinations(range(n_candidates), 2):
        m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)
    m.update()   

    #transitivity constraints
    for i, j, k in permutations(range(n_candidates), 3):
        m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k,i)] >= 1)
    m.update()

    #parity constraints
    parity = build_parity_constraints(groups)
    m.addConstr(quicksum(parity[i]*x[i] for i in range(len(x)))<= thresh)
    m.addConstr(quicksum(parity[i]*x[i] for i in range(len(x)))>= -thresh)

    # Set objective
    # maximize c.T * x
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()
    m.setObjective(quicksum(c[i]*x[i] for i in range(len(x))), GRB.MAXIMIZE)
    m.update()
    #m.write("kemeny_n"+str(n_ranks)+"_N"+str(rank_len) + "_t"+str(theta) + ".lp")
    t0 = time()
    m.optimize()
    t1 = time()

    if m.status == GRB.OPTIMAL:
        #m.write("kemeny_n"+str(rank_len)+"_N"+str(n_ranks)+"_t"+str(theta)+".sol")

        #get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol=np.array(sol)
        aggr_rank = np.sum(sol.reshape((n_candidates,n_candidates)), axis=1)
        return aggr_rank, t1-t0
    else:
        return None, t1-t0



def aggregate_equality(n_voters, n_candidates, ranks, groups, thresh):

    #Declare gurobi model object
    m = Model()
    m.setParam("OutputFlag", 0);

    # Indicator variable for each pair
    x = {}
    c=0
    for i in range(n_candidates):
        for j in range(n_candidates):
            x[c] = m.addVar(vtype=GRB.BINARY, name="x(%d)(%d)" %(i,j))
            c+=1
    m.update()

    idx = lambda i, j: n_candidates * i + j

    # pairwise constraints
    for i, j in combinations(range(n_candidates), 2):
        m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)
    m.update()   

    #transitivity constraints
    for i, j, k in permutations(range(n_candidates), 3):
        m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k,i)] >= 1)
    m.update()

    # equality constraints
    equality = all_pair_precedence(ranks, groups)
    flip = np.ones(len(x)) 
    
    m.addConstr(quicksum((flip[i] - x[i])*equality[i] for i in range(len(x)))<= thresh)
    m.addConstr(quicksum((flip[i] - x[i])*equality[i] for i in range(len(x)))>= -thresh)
    m.update()

    # Set objective
    # maximize c.T * x
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()
    m.setObjective(quicksum(c[i]*x[i] for i in range(len(x))), GRB.MAXIMIZE)
    m.update()
    #m.write("kemeny_n"+str(n_ranks)+"_N"+str(rank_len) + "_t"+str(theta) + ".lp")
    t0 = time()
    m.optimize()
    t1 = time()

    if m.status == GRB.OPTIMAL:
        #m.write("kemeny_n"+str(rank_len)+"_N"+str(n_ranks)+"_t"+str(theta)+".sol")

        #get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol=np.array(sol)
        aggr_rank = np.sum(sol.reshape((n_candidates,n_candidates)), axis=1)
        return aggr_rank, t1-t0
    else:
        return None, t1-t0
    
############################## Heuristics ######################################
#build matrix C to represent rankings to be aggregated
def copeland(ranks):
    t0 = time()
    n_voters, n_candidates = ranks.shape
    pair_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        pair_weights[i, j] = np.sum(preference > 0)  # prefers j to i
        pair_weights[j, i] = np.sum(preference < 0)  # prefers i to j
    sums = np.sum(pair_weights, axis=0)
    t1 = time()
    return np.argsort(sums), t1-t0

def borda(ranks):
    t0 = time()
    sums = np.sum(ranks, axis=0)
    t1 = time()
    return np.argsort(sums), t1-t0


""" The way our rankings are represented is with a vector of items 
where the index of each item corresponds to its item id
and the value at that index indicates the rank position of that item.
so groups[np.argsort(ranking)] yields the groups of the ranked items in ranked order.
We correct this, and then recover the representation of the ranking by calling 
np.argsort() on the vector f_agg, which is the ids of the items in rank order."""

def correct_parity(y, groups, thresh):
    ids = np.argsort(y)
    gs = groups[ids]
    f_agg = []
    #indices of items we skip
    skip = []
    #skip pointer to track when added back in
    ptr = 0 
    #get size of groups
    group_counts = np.bincount(groups)
    c0 = group_counts[0]
    c1 = group_counts[1]
    p0 = 0
    p1 = 0
    if (c0*c1) % 2 == 0:
        max_p = np.ceil(c0*c1/2) + np.ceil(thresh/2)
    else:
        max_p = np.ceil(c0*c1/2) + np.floor(thresh/2)
    
    for i,x in enumerate(gs):
        while ptr < len(skip):
            #check skipped items
            if gs[skip[ptr]] == 0 and p0 + c1 <= max_p: #and (p0 + c1) - (p1 +c0-1) <= max_p:
                p0 += c1
                c0 -= 1
                f_agg.append(ids[skip[ptr]])
                ptr = ptr+1
            elif gs[skip[ptr]] == 1 and p1 + c0 <= max_p: #and (p1 + c0) - (p0 +c1 - 1) <= max_p:
                p1 += c0
                c1 -= 1
                f_agg.append(ids[skip[ptr]])
                ptr = ptr + 1
            else:
                break
                            
        #handle next item
        if x == 0 and p0 + c1 <= max_p:
            p0 += c1
            c0 -= 1
            f_agg.append(ids[i])
        elif x == 1 and p1 + c0 <= max_p:
            p1 += c0
            c1 -= 1
            f_agg.append(ids[i])
        else:
            skip.append(i)
    #add in any remaining skipped items
    while ptr < len(skip):
        f_agg.append(ids[skip[ptr]])
        ptr = ptr + 1
    return np.argsort(f_agg)     
