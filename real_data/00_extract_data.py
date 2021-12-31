import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal

base = 'cut_5278/'
dat = [10, 20, 30, 40]
dat_2 = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]


idx = np.zeros((len(dat), len(dat_2), 25, 2024))
k=0
for j in dat:
    o = 0
    for yyy in dat_2:
        for i in range(1, 26):
            f = h5py.File(base + 'models/model_' + str(j) + str(yyy) + str(i) + '.out', 'r')
            idx[k, o, i-1, : ] = scipy.signal.resample(np.array(f['rxs']['rx1']['Ez'][:]), 2024)
        o += 1
    k += 1

np.save(base+'synth', idx)
