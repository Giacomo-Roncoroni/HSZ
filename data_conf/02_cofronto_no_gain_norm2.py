import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal


def moving_wind(dat, win_dims):
	return (scipy.signal.convolve(dat, np.ones(win_dims))/win_dims)[:dat.shape[0]]

path = 'data_no_gain/'
num  = '5278/'

nam_1 = [10, 12, 15, 17, 20, 22, 25, 27, 30, 35, 40, 45, 50, 55, 60]
nam_2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.375, 0.4]


data_synth = np.load(path + num + 'keep_' + num[:-1] + '.npy')
data       = np.load(path + num + 'data_cut.npy')
if  num == '5279/':
	data_ampli = np.append(0, np.load(path + num + 'base_trace.npy'))
else:
	data_ampli = np.load(path + num + 'base_trace.npy')

area_inter = np.load(path + num + 'area_interest.npy')

# create syth conf data
data_synth = (data_synth/np.max(data_synth))*np.max(data_ampli)
synt_ampli = np.sum(np.abs(data_synth), axis=2)/data_synth.shape[2]

# smute real
data_ampli[:int(area_inter[0, 1])] = 0
data_ampli[int(area_inter[0, 1] + area_inter[1, 1]):] = 0

#smute synth
synt_ampli[:, :, :int(area_inter[0, 1])] = 0
synt_ampli[:, :, int(area_inter[0, 1] + area_inter[1, 1]):] = 0

#apply window mean - real
real_conf = moving_wind(data_ampli, 100)

#apply window mean - synth
synt_conf = np.zeros(synt_ampli.shape)
for i in range(synt_ampli.shape[0]):
	for j in range(synt_ampli.shape[1]):
		synt_conf[i, j, :] = moving_wind(synt_ampli[i, j, :], 100)
		
out_conf = np.zeros((synt_conf.shape[0], synt_conf.shape[1]))
noormalized_synth = np.zeros(synt_conf.shape)
for i in range(synt_ampli.shape[0]):
	for j in range(synt_ampli.shape[1]):
		noormalized_synth[i, j, :] = (synt_conf[i, j, :]/np.max(synt_conf[i, j, :]))*np.max(real_conf)
		out_conf[i, j] = np.mean((real_conf - noormalized_synth[i, j, :])**2)


x, y = np.meshgrid(nam_1, nam_2)
z    = out_conf

min_idx = np.where(out_conf==np.min(out_conf))

plt.plot(noormalized_synth[min_idx[0][0], min_idx[1][0], :], 'k')
plt.plot(real_conf, 'r')
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x.T, y.T, z, cmap= 'viridis')
ax.set_xlabel('dimension [cm]')
ax.set_ylabel('lambda [nomalized]')
ax.set_zlabel('Energy')
ax.set_title('Energy')
plt.show()
plt.savefig('test_1_En', dpi = 500)