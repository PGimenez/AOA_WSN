import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('results.bin', 'rb') as f:
    mse = pickle.load(f)
    SNR = pickle.load(f)
    cfg = pickle.load(f)

print cfg
plt.plot(SNR, 10*np.log10(mse))
plt.grid(which='both')
plt.show()
