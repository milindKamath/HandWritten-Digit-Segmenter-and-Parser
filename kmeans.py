import numpy as np


class Kmeans:

    def __init__(self, k=0):
        self.k = k
        self.feat = []
        self.clusters = {}

    def kmeans(self, feat):
        centroids = feat[np.random.randint(feat.shape[0], size=self.k), :]
        counter = 1
        while True:
            clust = {}
            for i in range(feat.shape[0]):
                clID = np.argmin(np.sqrt(np.sum(np.square(np.subtract(centroids, feat[i])), axis=1)))
                clust.setdefault(clID, []).append(i)
            newCent = np.zeros(centroids.shape)
            for cId in clust:
                newCent[cId] = np.sum(clust[cId], axis=0, keepdims=True)/len(clust[cId])
            breakCondition = np.abs(np.subtract(centroids, newCent))
            centroids = newCent
            counter += 1
            if not np.count_nonzero(breakCondition) or counter == 100:
                return clust
