from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def complementary_angle(angle):
    angle[angle>np.pi] = angle[angle>np.pi] - 2 * np.pi
    angle[angle<-np.pi] = angle[angle<-np.pi] + 2 * np.pi
    return angle

def LOS_probability(angle_error, variance, p):
    exponent = -angle_error / (2.0 * variance)
    exponent[exponent<-700] = -700
    rho_1 = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(exponent)
    return rho_1 * p / ((0.5 / np.pi)  * (1-p) + rho_1 * p)

def clairvoyant_LS(sensors, AOA, distance):
    b = -sensors.node_positions[:,0] * np.sin(AOA) + sensors.node_positions[:,1] * np.cos(AOA)
    H = np.c_[-np.sin(AOA), np.cos(AOA)]
    V = np.diag((1 / distance) * sensors.LOS)
    return np.linalg.inv(H.T.dot(V).dot(H)).dot(H.T).dot(V).dot(b)

def centralized_EM(sensors, AOA, p_0, iterations):
    #calculate the initial estimates for x,y,variance and p
    b = -sensors.node_positions[:,0] * np.sin(AOA) + sensors.node_positions[:,1] * np.cos(AOA)
    H = np.c_[-np.sin(AOA), np.cos(AOA)]
    V = 1
    mt_position = np.linalg.inv(H.T.dot(V).dot(H)).dot(H.T).dot(V).dot(b)
    AOA_estimate, distance = sensors.calculate_AOA(mt_position)
    angle_error = complementary_angle(AOA - AOA_estimate)**2
    p = p_0
    a = p_0 * np.ones(sensors.N)
    variance = np.sum(a*angle_error) / np.sum(a)
    V = np.diag((1 / distance) * a)
    #enter the EM loop
    for k in range(0, iterations):
        #expectation step
        a = LOS_probability(angle_error, variance, p)
        #maximization step
        mt_position = np.linalg.inv(H.T.dot(V).dot(H)).dot(H.T).dot(V).dot(b)
        AOA_estimate, distance = sensors.calculate_AOA(mt_position)
        angle_error = complementary_angle(AOA - AOA_estimate)**2
        V = np.diag((1 / distance) * a)
        variance = np.sum(a*angle_error) / np.sum(a)
        p = np.mean(a)
        #if variance < 0.001:
            #break
    return mt_position


def distributed_EM(sensors, AOA, p_0, delta, iterations):
    def intermediate_vars(PHI, sensors, AOA, angle_error, a, distance, k, delta):
        F = np.zeros((sensors.N, 8))
        F[:,0] = a
        F[:,1] = np.ones((sensors.N))
        F[:,2] = distance*a*np.sin(AOA)**2
        F[:,3] = -distance*a*np.sin(AOA)*np.cos(AOA)
        F[:,4] = distance*a*np.cos(AOA)**2
        F[:,5] = -distance*a*np.sin(AOA)*(-sensors.node_positions[:,0]*np.sin(AOA) + sensors.node_positions[:,1]*np.cos(AOA))
        F[:,6] = distance*a*np.cos(AOA)*(-sensors.node_positions[:,0]*np.sin(AOA) + sensors.node_positions[:,1]*np.cos(AOA))
        F[:,7] = a*angle_error
        beta = 1.0/k**delta
        gamma = 1.0/k
        return (1-beta) * sensors.W.dot(PHI) + gamma * sensors.W.dot(F);

    def update_estimates(PHI):
        #mat1 = np.c_[PHI[:,2], PHI[:,3],PHI[:,3], PHI[:,4]].reshape(sensors.N, 2, 2)
        #mat1 = np.linalg.inv(mat1)
        #mat2 = np.c_[PHI[:,5], PHI[:,6]].reshape(sensors.N, 2, 1)
        #mt_position = np.sum(np.transpose(mat1,(0,2,1)).reshape(sensors.N,2,2,1) * mat2.reshape(sensors.N,2,1,1),-3)
        #mt_position = mt_position.reshape(sensors.N,2)
        mt_position = np.zeros((sensors.N,2))
        for i in range(0,sensors.N):
            mat1 = np.array([[PHI[i,2], PHI[i,3]],[PHI[i,3], PHI[i,4]]])
            #use pinv in case mat1 is singular.....
            mat1 = np.linalg.pinv(mat1)
            mat2 = np.array([[PHI[i,5]], [PHI[i,6]]])
            mt_position[i,:] = mat1.dot(mat2).T
        p = PHI[:,0] / PHI[:,1]
        variance = PHI[:,7] / PHI[:,0]
        return mt_position, variance, p

    PHI = np.zeros((sensors.N, 8))
    a = np.zeros((sensors.N,iterations))
    mt_position = np.zeros((iterations,sensors.N,2))
    a[:,0] = p_0 * np.ones(sensors.N)
    PHI = intermediate_vars(PHI, sensors, AOA, 5, a[:,0], 1, 1, 0)
    mt_position[0,:,:], variance, p = update_estimates(PHI)
    AOA_estimate, distance = sensors.calculate_AOA(mt_position[0,:,:])
    angle_error = complementary_angle(AOA - AOA_estimate)**2
    for k in range(0, iterations):
        a[:,k] = LOS_probability(angle_error, variance, p)
        a[a<1e-8] = 1e-8
        variance[variance<1e-8] = 1e-8
        PHI = intermediate_vars(PHI, sensors, AOA, angle_error, a[:,k], 1/distance, k+1, delta)
        mt_position[k,:,:], variance, p = update_estimates(PHI)
        AOA_estimate, distance = sensors.calculate_AOA(mt_position[k,:,:])
        angle_error = complementary_angle(AOA - AOA_estimate)**2
        if np.mean(variance) < 1e-15:
            break

        #print k
        #print a
        #print angle_error
    #plt.figure()
    #plt.plot(a.T)
    #plt.title('delta = ' + `delta`)
    #plt.figure()
    #plt.plot(mt_position[:,:,0])
    #plt.title('delta = ' + `delta`)
    #plt.show()
    #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    return np.mean(mt_position[-1,:,:],0)


