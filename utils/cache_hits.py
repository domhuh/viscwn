import numpy as np
import random
from itertools import chain, combinations
from collections import defaultdict
from functools import partial

def indicate(f,S):
    flag = 0
    for sbs in S:
        if sbs.fetch(f):
            sbs.hits+=1
            flag=1
        else:
            sbs.misses+=1
    return flag

def make_zipf(n_content,lam):
    l = 0
    for j in range(1,n_content+1): l += j**-lam
    f_i = [(i**-lam)/l for i in range(1,n_content+1)]
    return f_i
    
def get_fi(gamma,L,i):
    l = 0
    for j in range(1,L+1): l += j**-gamma
    return (i**-gamma)/l

def powerset(s):
    return list(chain.from_iterable(list(combinations(s, r)) for r in range(len(s)+1)))

def check_file(f,S):
    for sbs in S:
        if sbs.fetch(f):
            return 1
    return 0

def calculate_chp(network):
    chp = 0.
    for user in network.users:
        Z = powerset(user.connected_sbs)[1:]
        if len(Z)==0:
            continue
        x = [1/len(Z)*sum([check_file(f,S) for S in Z]) for f in user.profile]
        chp += np.dot(user.f_i,x)
    return chp/len(network.users), None

def calculate_expected_chp(network):
    requested_data = defaultdict(partial(defaultdict,int))
    chp = 0.
    for user in network.users:
        user_chp = 0.
        for i in range(user.request_freq):
            tmp, data= user.request()
            user_chp += tmp
            for sbs in user.connected_sbs:
                requested_data[sbs.id][data] += 1
        chp += user_chp/user.request_freq
    return chp/len(network.users), requested_data