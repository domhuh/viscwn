import numpy as np
from env.network import BaseNetwork
from env.basecenter import BaseCenter

class SINRNetwork(BaseNetwork):
    def __init__(self, user_lam=0.2887/2, sbs_lam=0.2887/2,
                 n_sbs=None, n_user=None, n_content=1000,
                 approximation=True):
        super(SINRNetwork,self).__init__(user_lam, sbs_lam, n_sbs, n_user, n_content, approximation)        
        self.coverage_model = 'sinr'
        self.coverage_threshold = 15 #dB
        self.w = 3.7 #meters (U.S. standard lanes for IHS)
        self.H_roof = 3.5 #meters (average for mixed buildings)
        self.b = 3.048 #meters (min building separations)
        self.center_type = 'metro'
        self.reset(hard_reset=True)
        self.init_vis=True
        self.id_hm = {sbs.id:sbs for sbs in self.sbs}
    def theta(self,user,sbs):
        return angle_between((sbs.x,sbs.y,self.H_roof),(user.x,user.y,user.z))
    def create_sbs(self,x,y,cache_size,id_):
        return BaseCenter(x,y,1,
                          cache_size,id_,self.sbs_lam)
    
def sample_spherical(rad, ndim=2):
    vec = np.random.randn(ndim, 1)*rad
    vec /= np.linalg.norm(vec, axis=0)
    return vec
def unit_vector(vector):
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def pathwise_loss_WI(user,sbs,network,d=None,**kwargs):
    d = user.distance_from(sbs) if d is None else d #meter to km
    if d<0.02:
        return 1.
    Lo = 32.45 + 20*np.log10(d) + 20*np.log10(sbs.freq)
    Lrts = 0
    def get_L_cri(theta):
        if theta>90:
            return 0
        elif theta>55:
            return 4. - 0.114*(theta-55)
        elif theta>35:
            return 3. - 0.075*(theta-35)
        else:
            return -10+0.354*theta
    if sbs.z>=user.z:
        L_cri = get_L_cri(np.degrees(network.theta(sbs,user)))
        Lrts = -16.9 - 10*np.log10(network.w) + 10*np.log10(sbs.freq) + 20*np.log10(network.H_roof - user.z) + L_cri
    def get_Lbsh(H_b,H_roof):
        if H_b>H_roof:
            return -18*np.log10(1+H_b-H_roof)
        return 0
    def get_ka(H_b,H_roof,d):
        if H_b==H_roof: return 54.
        if H_b<=H_roof:
            if d>=0.5:
                return 54. - 0.8*(H_b-H_roof)
            else:
                return 54. - 0.8*(H_b-H_roof)*(d/0.5)
        return 0
    def get_kd(H_b,H_roof):
        if H_b>H_roof:
            return 18
        return 18 - 15*(H_b-H_roof)/H_roof
    def get_kf(network,sbs):
        if network.center_type =='suburban':
            return -4+0.7*(sbs.freq/925 -1)
        return -4+1.5*(sbs.freq/925 -1)
    Lbsh = get_Lbsh(sbs.z, network.H_roof)
    ka = get_ka(sbs.z, network.H_roof, d)
    kd = get_kd(sbs.z, network.H_roof)
    kf = get_kf(network,sbs)
    Lmsd = Lbsh + ka + kd * np.log10(d) + kf * np.log10(sbs.freq) + 9*np.log10(network.b)
    return Lo+Lrts+Lmsd

def f(user,network,s):
    return np.random.rayleigh(1)*np.power(10,np.clip(s.S-pathwise_loss_WI(user,s,network),-100,100),dtype=np.float64)

from multiprocessing import Pool,cpu_count
from functools import partial
def get_values_WI(sbs,network,x,y):
    from user import User    
    W = network.W
    user = User(x=x,y=y)
    l = pathwise_loss_WI(user,sbs,network)
    S = sbs.S-l
    r_sbs = np.delete(np.copy(network.sbs),np.where(network.sbs == sbs)) 
    with Pool(cpu_count()) as p:
        I = p.map(partial(f,user,network), r_sbs)
        I = I[:]
    I_ = np.log10(sum(I))
    W = network.W #Noise (dbm)
    return S, I_, W, S-np.log10(sum(I)+10**W)
class VIS(SINRNetwork):
    def __init__(self, n_sbs):
        super(VIS,self).__init__(n_sbs=n_sbs)
        self.n_sbs = n_sbs
    def populate_users(self):
        return
    def get_image(self):
        c1 = np.zeros(shape=(1,200,200))
        for sbs in self.sbs:
            c1[0,int(sbs.x*10), int(sbs.y*10)] = sbs.S
        return c1
    def get_metadata(self,x,y):
        sinr = 0
        for sbs in self.sbs:
            sinr+=np.power(10,get_values_WI(sbs,self,x,y)[-1])        
        return np.log10(sinr) #SINR
    def toJSON(self):
        import json
        from json_tools import JSONable
        from copy import deepcopy
        data = deepcopy(self.__dict__)
        data['users'] = []
        data['sbs'] = [sbs.toJSON() for sbs in data['sbs']]
        data['action_space'] = data['action_space'].toJSON()
        return json.dumps(data, cls=JSONable)

if __name__ == '__main__':
    import torch
    from sklearn.model_selection import train_test_split
    import os
    from tqdm import tqdm
    n=8 #ran on eight cores (runs fastest)
    for _ in tqdm(range(int(1e5))):
        for k in range(2,100):
            filename = f"{_}{k}"
            env = VIS(n_sbs=k)
            env.save_network(f"{filename}.json")
            metadata = []
            for l in range(n):
                x_ = np.random.uniform(-10,10)
                y_ = np.random.uniform(-10,10)
                sinr = env.get_metadata(x_,y_)
                metadata.append([x_,y_,sinr])
            img = env.get_image()
            img = torch.FloatTensor(img)
            metadata = torch.FloatTensor(metadata)
            train_indices,test_indices = train_test_split(np.arange(n),train_size=0.8)
            torch.save(img,f"{filename}_image.pt")
            torch.save(metadata[train_indices],f"{filename}_metadata_train.pt")
            torch.save(metadata[test_indices],f"{filename}_metadata_test.pt")

    for _ in tqdm(range(10)):
        for k in range(2,100):
            filename = f"NOTSEEN_{_}{k}"
            env = VIS(n_sbs=k)
            env.save_network(f"{filename}.json")
            metadata = []
            for l in range(n):
                x_ = np.random.uniform(-10,10)
                y_ = np.random.uniform(-10,10)
                sinr = env.get_metadata(x_,y_)
                metadata.append([x_,y_,sinr])
            img = env.get_image()
            img = torch.FloatTensor(img)
            metadata = torch.FloatTensor(metadata)
            torch.save(img,f"{filename}_image.pt")
            torch.save(metadata,f"{filename}_metadata_test.pt")


