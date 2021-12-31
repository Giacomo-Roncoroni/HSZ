import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal


def moving_wind(dat, win_dims):
	return (scipy.signal.convolve(dat, np.ones(win_dims))/win_dims)[:dat.shape[0]]

num  = '5278/'

#dat = [10, 15, 20, 25, 30, 35, 40, 45, 50]
dat_1 = [10, 20, 30, 40]
dat_2 = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

data_synth = np.load('cut_' + num[:-1] + '/synth.npy')


synt_ampli2 = np.zeros(data_synth.shape)

for i in range(len(dat_1)):
	for j in range(len(dat_2)):
		for k in range(25):
			synt_ampli2[i, j, k, :] = moving_wind(np.abs(data_synth[i, j, k, :]), 32)

print(data_synth.shape)
synt_ampli = np.mean(np.abs(data_synth), axis=2)/data_synth.shape[2]
synt_ampli2 = np.mean(np.abs(synt_ampli2), axis=2)/synt_ampli2.shape[2]

synt_ampli[:, :, :250] = 0
synt_ampli2[:, :, :250] = 0

plt.title('Trend conf')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
#for i in range(len(dat_1)):
#for j in range(len(dat_2)):
for i in [0, 1, 2, 3]:
	for j in [4]:
		plt.plot(moving_wind(synt_ampli2[i, j, :], 32)/np.max(moving_wind(synt_ampli2[i, j, :], 32)), label='dims:' + str(dat_1[i]) + ' lamb: ' + str(dat_2[j]), linewidth = 0.25)

plt.legend()
plt.savefig('Input_wave' + num[:-1], dpi=300)
plt.clf()

