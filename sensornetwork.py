
import numpy as np
import matplotlib.pyplot as plt

class SensorNetwork():

    def __init__(self, N, dim_x, dim_y, radius, rule, p_LOS):
        self.N = N
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.radius = radius
        self.A = np.zeros((N,N))
        self.gen_graph()
        self.gen_weights(rule)
        self.LOS = np.around(p_LOS + np.random.uniform(-0.5,0.5,N))
        self.NLOS = 1 - self.LOS
        #self.NLOS = np.array([1,0,0,0])
        #self.LOS = np.array([0,1,1,1])

    def calculate_AOA(self, mt_position, std = 0, measurement = False):
        distance_xy = mt_position - self.node_positions
        distance = np.sqrt(np.sum(distance_xy**2,1))
        AOA = np.arctan2(distance_xy[:,1], distance_xy[:,0]) + std*np.random.randn(self.N)
        if measurement:
            AOA[self.NLOS.astype(bool)] = np.random.uniform(-1, 1, np.sum(self.NLOS)) * np.pi
        return AOA, distance

    def gen_graph(self):
        self.node_positions = np.random.uniform(0, 1, (self.N,2)) * [self.dim_x, self.dim_y]
        #node_positions = np.zeros((N,2))
        #node_positions[:,0] = np.random.uniform(0, dim_x, N)
        #node_positions[:,1] = np.random.uniform(0, dim_y, N)
        r = self.radius**2
        for i in range(0,self.N):
            for j in range(i+1,self.N):
                distance = np.sum((self.node_positions[i,:] - self.node_positions[j,:])**2)
                if distance < r:
                    self.A[i,j] = 1
                    self.A[j,i] = 1

    def gen_weights(self, rule):
        if rule == 'metropolis':
            degree = self.A.dot(np.ones(self.N))
            self.W = np.zeros((self.N,self.N))
            for i in range(0, self.N):
                for j in range(i+1, self.N):
                    if self.A[i,j]:
                        self.W[i,j] = 1 / (1 + np.max([degree[i],degree[j]]))
            self.W = self.W + self.W.T
            self.W = self.W + np.eye(self.N) - np.diag(self.W.dot(np.ones(self.N)))
        elif rule == 'full':
            self.W = np.ones((self.N,self.N)) / self.N

