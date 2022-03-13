from collections import defaultdict
from functools import partial
from operator import itemgetter
import numpy as np

def update_stat(stat,pop,fi):
    for file_,freq in zip(pop,fi):
        stat[file_] += freq

class MPC_Policy():
    def __init__(self,env,oracle=False):
        self.env = env
        self.sbs = self.env.sbs
        self.cache_hashmap = {sbs.id:sbs.cache_size for i,sbs in enumerate(self.sbs)}
        self.oracle = oracle
        if not self.oracle:
            self.reset()
        else:
            self.precompute_action()
        self.gamma = 0.99
    def reset(self):
        self.ref = defaultdict(partial(defaultdict,float))
        for sbs in self.sbs:
            n = 0
            for user in sbs.connected_users:
                if n<user.n_content: n = user.n_content
            self.ref[sbs.id] = {c:0 for c in range(n)}
        self.update({})
        
    def update(self,state):
        if state is None or self.oracle: return
        for sbs_id,requests in state.items():
            for request_id, n_requests in requests.items():
                self.ref[sbs_id][request_id] += (1-self.gamma)*self.ref[sbs_id][request_id]+self.gamma*n_requests
        stat=defaultdict(partial(defaultdict,list))
        for sbs_id,content in self.ref.items():
            stat[sbs_id] = [k for k,v in sorted(content.items(), key=itemgetter(1), reverse=True)][:self.cache_hashmap[sbs_id]]
            stat[sbs_id] = np.pad(stat[sbs_id],[0,self.cache_hashmap[sbs_id]-len(stat[sbs_id])])
        self.action = [stat[sbs.id] for sbs in self.sbs]
                
    def precompute_action(self):
        stat=defaultdict(partial(defaultdict,list))
        for sbs in self.sbs:
            tmp = defaultdict(float)
            for user in sbs.connected_users:
                update_stat(tmp,user.profile,user.f_i)
            stat[sbs.id] = [k for k,v in sorted(tmp.items(), key=itemgetter(1), reverse=True)][:sbs.cache_size]
            stat[sbs.id] = np.pad(stat[sbs.id],[0,sbs.cache_size-len(stat[sbs.id])])
        self.action = [stat[sbs.id] for sbs in self.sbs]

    def __call__(self):
        return self.action