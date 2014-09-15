import numpy as np

def centralized_EM(node_positions, AOA, K):
    b = -node_positions[:,0]*np.sin(AOA) + node_positions[:,1]*np.cos(AOA)
    H = np.array([-np.sin(AOA), np.cos(AOA)])
    p = np.zeros((K,1))
    var = np.zeros((K,1))
    a = np.zeros((AOA.shape[0],K))
    a = np.zeros((2,K))
    W = 1
    rho_0 = 0.5*pi
    # EM initialization
    r[:,0] = np.inv(H.T.dot(H)).dot(H.T).dot(b)
    distance_xy = r[:,0].T - node_positions
    AOA_est = np.arctan2(distance_xy[:,0], distance_xy[:,1])
    angle_error = AOA - AOA_est
    #unroll
    var[0] = np.var(AOA)
    p[0] = 0.5
    for i in range(0,K-1):
        # EXPECTATION
        rho_1 = 1 / np.sqrt(2 * pi * var(k) * np.exp(-0.5 * angle_error / var(i)))
        a[:,i] = rho_1 * p(i) / ((rho_0 * (1-p(i))) + rho_1 * p(i))
        # MAXIMIZATION
        distance = np.sqrt(np.sum((r[:,0].T - node_positions)**2,1))
        W = np.diag((1 / distance) * a)
        r[:,i] = np.linalg.inv(H.T.dot(W).dot(H)).dot(H.T).dot(W).dot(b)
        distance_xy = r[:,i].T - node_positions
        AOA_est = np.arctan2(distance_xy[:,0], distance_xy[:,1])
        angle_error = AOA - AOA_est
        var[i] = a[:,i].dot(angle_error) / np.sum(a[:,i])
