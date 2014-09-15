from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sensornetwork
from estimation import *
import config as cfile
import pickle
from multiprocessing import Pool

def run_simulation_SNR(cfg):
    #mse = np.zeros(((SNR_max - SNR_min) / SNR_step + 1, 3))
    #SNR_vec = np.arange(SNR_min, SNR_max, SNR_step)
    #for i,SNR in enumerate(SNR_vec):
    N, dim_x, dim_y, radius, rule, p_LOS, delta, realizations, iterations, SNR = map(cfg.get,('N', \
        'dim_x', 'dim_y', 'radius', 'rule', 'p_LOS', 'delta', 'realizations', 'iterations', 'SNR'))
    print SNR
    error = np.zeros((realizations, 3))
    for r in np.arange(0, realizations):
        sensors = sensornetwork.SensorNetwork(N, dim_x, dim_y, radius, rule, p_LOS)
        #sensors = sensornetwork.SensorNetwork(4, dim_x, dim_y, radius, rule, p_LOS)
        #sensors.node_positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]]);
        mt_position_real = np.random.uniform(0,1,2)* np.array([dim_x, dim_y])
        #mt_position_real = [100,100]
        #print "real position" + `mt_position_real`
        AOA, _ = sensors.calculate_AOA(mt_position_real, np.sqrt(10**(-SNR/10)), measurement=True)
        #AOA = np.array([ 2.78178438,  1.76218996, -1.61948646, -4.77524095])
        #AOA, _ = sensors.calculate_AOA(mt_position_real, measurement = True)
        #AOA[0] = -np.pi/2
        # Clairvoyant LS
        mt_position = clairvoyant_LS(sensors, AOA, 1)
        _, distance = sensors.calculate_AOA(mt_position)
        mt_position = clairvoyant_LS(sensors, AOA, distance)
        error[r,0] = np.sqrt(np.sum(mt_position - mt_position_real)**2)
        # Centralized EM
        mt_position = centralized_EM(sensors, AOA, p_LOS, iterations)
        error[r,1] = np.sqrt(np.sum(mt_position - mt_position_real)**2)
        # Distributed EM
        mt_position = distributed_EM(sensors, AOA, p_LOS, delta, iterations)
        error[r,2] = np.sqrt(np.sum(mt_position - mt_position_real)**2)
    return np.mean(error,0)

for cfg in cfile.confdicts:
    SNR_vec = np.arange(cfg['SNR_min'], cfg['SNR_max'], cfg['SNR_step'])
    mse = np.zeros((SNR_vec.shape[0], 3))
    #cfg = {x:config.__dict__[x] for x in dir(config) if '__' not in x}
    cfg_list = [dict(cfg,SNR=val) for val in SNR_vec]
    print cfg
    #for i, conf in enumerate(cfg_list):
        ##np.random.seed(10)
        #mse[i] = run_simulation_SNR(conf)
    #run in parallel
    pool = Pool(processes=12)
    for i, result in enumerate(pool.imap(run_simulation_SNR, cfg_list, 1)):
        np.random.seed(10)
        mse[i] = result
        pool.close()
    print mse
    #save plot
    filename = 'R_'+ `cfg['radius']` + '_iter_'+ `cfg['iterations']` + '_delta_'+ `cfg['delta']`
    plt.plot(SNR_vec, 10*np.log10(mse))
    plt.grid(which='both')
    plt.title('R = ' + `cfg['radius']` + ' delta = ' + `cfg['delta']`)
    plt.savefig('fig/' + filename + '.pdf')
    plt.close()
    #save results
    with open('results/'+ filename + '.bin', 'wb') as f:
        pickle.dump(mse, f)
        pickle.dump(SNR_vec, f)
        pickle.dump(cfg, f)
