import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import signal

nam_1 = [10, 12, 15, 17, 20, 22, 25, 27, 30, 35, 40, 45, 50, 55, 60]
nam_2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.375, 0.4]

iter_n = 5
t_len = 2048

idx = np.zeros((len(nam_1), len(nam_2), iter_n))
keep = np.zeros((len(nam_1), len(nam_2), iter_n, t_len))
k=0

for i in range(len(nam_1)):
    for j in  range(len(nam_2)):
        for k in range(iter_n):
            f = h5py.File('models/model_' + str(nam_1[i]) + str(nam_2[j])  + str(k+1) + '.out', 'r')
            dat = np.array(f['rxs']['rx1']['Ez'][:])
            idx[i, j, k] = np.sum(np.abs(signal.resample(dat, t_len)))
            #keep[i, j, k, :] = signal.resample(dat, t_len)
            #if j == 0.9:
            #    np.save('test_trace_' + str(i), np.array(f['rxs']['rx1']['Ez'][:]))
    k += 1


x, y = np.meshgrid(nam_1, nam_2)
z    = np.mean(idx, axis=-1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.contour3D(x.T, y.T, z, 50, cmap= 'viridis')
ax.set_xlabel('dimension [cm]')
ax.set_ylabel('lambda [nomalized]')
ax.set_zlabel('Energy')
ax.set_title('Energy')


ax.view_init(20, 10)

plt.savefig('test_1_En', dpi = 500)
plt.clf()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_wireframe(x.T, y.T, z,  cmap= 'viridis')
ax.set_xlabel('dimension [cm]')
ax.set_ylabel('lambda [nomalized]')
ax.set_zlabel('Energy')
ax.set_title('Energy')


ax.view_init(20, 10)
plt.savefig('test_2_En', dpi = 500)
plt.clf()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(x.T, y.T, z, c=z, cmap='viridis', linewidth=0.5)
ax.set_xlabel('dimension [cm]')
ax.set_ylabel('lambda [nomalized]')
ax.set_zlabel('Energy')
ax.set_title('Energy')

ax.view_init(20, 10)
plt.savefig('test_3_En', dpi = 500)
plt.clf()

plt.contour(x.T, y.T, z, 50, cmap='viridis', linewidth=0.5)
plt.xlabel('dimension [cm]')
plt.ylabel('lambda [nomalized]')
plt.colorbar()
plt.title('Energy')

plt.savefig('test_4_En', dpi = 500)

np.save('final_cut', keep)
