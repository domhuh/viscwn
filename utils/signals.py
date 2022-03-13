import numpy as np

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

def pathwise_loss(user,sbs,beta=3.8,K=1,d=None,**kwargs):
    assert beta>=2
    if d is not None:
        return (K*d)**-beta
    return (K*user.distance_from(sbs)+1e-6)**-beta

def sinr(user,sbs,network,d=None):
    if d is not None:
        l = pathwise_loss(user,sbs,d=d)
    else:
        l = pathwise_loss(user,sbs)
    r_sbs = np.delete(network.sbs,np.where(network.sbs == sbs)) #remove current sbs from interference calculation
    I = [np.random.rayleigh(1)*np.power(10,np.clip(sbs.S*pathwise_loss(user,sbs),-100,100),dtype=np.float64)] #Interference/Shot-noise process with Rayleigh fading
    W = network.W #Noise (dbm)
    S = sbs.S*l
    return S-np.log10(sum(I)+10**W)

def sinr_WI(user,sbs,network,d=None):
    if d is not None:
        l = pathwise_loss_WI(user,sbs,network=network,d=d)
    else:
        l = pathwise_loss_WI(user,sbs,network=network)
    r_sbs = np.delete(network.sbs,np.where(network.sbs == sbs)) #remove current sbs from interference calculation
    I = [np.random.rayleigh(1)*np.power(10,s.S-pathwise_loss_WI(user,s,network=network)) for s in r_sbs] #Interference/Shot-noise process with Rayleigh fading
    W = network.W #Noise (dbm)
    S = sbs.S-l
    return S-np.log10(sum(I)+10**W)

def get_values(sbs,network,x,y):
    from env.user import User    
    W = network.W
    user = User(x=x,y=y)
    l = pathwise_loss(user,sbs)
    S = sbs.S-l
    r_sbs = np.delete(network.sbs,np.where(network.sbs == sbs)) #remove current sbs from interference calculation
    I = [np.random.rayleigh(1)*np.power(10,np.clip(sbs.S*pathwise_loss(user,sbs),-100,100),dtype=np.float64)] #Interference/Shot-noise process with Rayleigh fading
    I_ = np.log10(sum(I))
    W = network.W #Noise (dbm)
    S = sbs.S*l
    return S, I_, W, S-np.log10(sum(I)+10**W)

def get_values_WI(sbs,network,x,y):
    from env.user import User
    W = network.W
    user = User(x=x,y=y)
    l = pathwise_loss_WI(user,sbs,network=network)
    S = sbs.S-l
    r_sbs = np.delete(network.sbs,np.where(network.sbs == sbs)) #remove current sbs from interference calculation
    I = [np.random.rayleigh(1)*np.power(10,s.S-pathwise_loss_WI(user,s,network=network)) for s in r_sbs] #Interference/Shot-noise process with Rayleigh fading
    I_ = np.log10(sum(I))
    return S, I_, W, S-np.log10(sum(I)+10**W)