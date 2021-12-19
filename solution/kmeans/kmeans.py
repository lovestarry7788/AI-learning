import numpy as np
from sklearn import datasets
import pdb
import math
import matplotlib.pyplot as plt

class KMeansBase:

    def __init__(self, n_cluster = 8, max_iter = 3000):
        self.k = n_cluster
        self.max_iter = max_iter
    
    def _init_centroids(self, dataset):
        m, n = dataset.shape
        centroids = np.zeros((self.k,n))
        for i in range(self.k):
            idx = np.random.randint(m)
            centroids[i] = dataset[idx]
        return centroids

    def fit(self, dataset):
        dataset = np.array(dataset)
        print(dataset.shape)
        centroids = self._init_centroids(dataset)
        k = self.k
        m, n = dataset.shape

        plt_t = []
        plt_dis = []

        for t in range(self.max_iter):
            count = np.zeros((k))
            means = np.zeros((k,n))
            distance = 0.
            for i in range(len(dataset)):
                minDist = 1e9
                minIndex = 0
                for j in range(len(centroids)):
                    dis = np.sqrt(np.sum(np.square(dataset[i] - centroids[j])))
                    if dis < minDist:
                        minDist = dis
                        minIndex = j
                if minDist != 1e9:
                    
                    count[minIndex] = count[minIndex] + 1
                    means[minIndex] = means[minIndex] + dataset[i]
                    distance = distance + minDist
            
            print("Iter numbers : %f , Distance : %f , Cluster Number %d"%(t,distance,self.k))
            print("centroids :")
            print(centroids)
            for i in range(len(centroids)):
                if count[i]:
                    centroids[i] = means[i] / count[i]
            print()
            plt_t.append(t)
            plt_dis.append(distance)

        plt.plot(plt_t,plt_dis,label = 'Cluster Number: %d'%(k))
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        
        

                


if __name__ == '__main__':
    dataset = datasets.load_iris()
    for k in range(2,50) :
        km = KMeansBase(k,20)
        km.fit(dataset.data)

    plt.legend()
    plt.show()

'''
load_iris()
'''