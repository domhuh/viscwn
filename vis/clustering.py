from sklearn.cluster import KMeans
import numpy as np
def cluster(sbs,network):
    data = []
    chp = []
    for user in sbs.connected_users:
        if user in network.users:
            data.append((user.x,user.y))
            chp.append(user.chp)

    if len(data)<4:
        return data, np.arange(len(data)), data, chp

    #elbow
    scores=[]
    for i in range(1,min(10,len(data)-1)):
        cluster = KMeans(n_clusters=i, random_state=0)
        cluster.fit(data)
        scores.append(cluster.inertia_)
    print(scores)
    n=np.argmax(np.diff(scores)**2<1e1)+1
    #check for no end
    if type(n)!=np.int64:
        n=len(data)
    print(n)
    cluster = KMeans(n_clusters=n, random_state=0)
    cluster.fit(data)
    return data, cluster.labels_, cluster.cluster_centers_, chp