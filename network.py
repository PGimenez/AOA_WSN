
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class SensorNetwork():

    def __init__(self, N, dim_x, dim_y, radius, rule, p_LOS):
        self.N = N
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.radius = radius
        self.A = np.zeros((N,N))
        self.gen_graph()
        self.gen_weights(rule)
        self.NLOS = np.round(p_LOS + np.random.uniform(0,1,(N,1)))

    def measure_AOA(self, mt_position, node_positions, ind_NLOS, std, noise = 1):
        distance_xy = mt_position - node_positions
        distance = np.sqrt(np.sum(distance_xy**2,1))
        AOA = np.arctan2(distance_xy[:,0], distance_xy[:,1]) + noise * std*np.random.randn(N)
        AOA[ind_NLOS] = np.random.uniform(0, 1, ind_NLOS.shape) * 2 * pi
        return AOA

    def gen_graph(self):
        self.node_positions = np.random.uniform(0, 1, (self.N,2)) * [self.dim_x, self.dim_y]
        #node_positions = np.zeros((N,2))
        #node_positions[:,0] = np.random.uniform(0, dim_x, N)
        #node_positions[:,1] = np.random.uniform(0, dim_y, N)
        r = self.radius**2
        for i in range(0,N-1):
            for j in range(i+1,N-1):
                distance = np.sum((self.node_positions[:,i] - self.node_positions[:,j])**2)
                if distance < r:
                    self.A[i,j] = 1
                    self.A[j,i] = 1

    def gen_weights(self, rule):
        if rule == 'metropolis':
            degree = self.A.dot(np.ones(self.N))
            self.W = np.zeros((N,N))
            for i in range(0, self.N):
                for j in range(i+1, self.N):
                    if self.A[i,j]:
                        self.W[i,j] = 1 / (1 + np.max(degree[i],degree[j]))
            self.W = self.W + self.W.T
            self.W = self.W + np.eye(self.N) - np.diag(self.W*np.ones(self.N))

