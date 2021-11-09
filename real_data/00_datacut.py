import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
import warnings
import matplotlib.cbook
from scipy.interpolate import interp1d
import matplotlib.patches as patches
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import scipy.signal

# read segy file 
def read_segy(path):
	# test all type of endian
	for i in ['big', 'little']:
		# test an endian
		try: segyio.open(path, endian = i, ignore_geometry= True)
		# if error: change endian (i) and retry
		except RuntimeError:
			print('Probably a wrong endian format was used! Try with little')
		# if not errorm exit the loop --> i is the correct endian
		else:
			break
	# read the data
	s=segyio.open(path, endian = i, ignore_geometry= True)
	# init the matrix
	mat=np.zeros([s.trace.length, s.trace.shape])
	# fill the matrix with values for each trace
	for i in range(s.trace.length):
		mat[i,:] = s.trace[i]
	print('Data succesfully loaded!')
	return mat


#plot function
def plot_horiz(new_matrix):
	plt.subplot(121)
	plt.imshow(new_matrix, aspect='auto')
	plt.subplot(122)
	for i in range(n_horiz):
		plt.plot(sorted_horiz[i, :, 0], sorted_horiz[i, :, 1])
		plt.xlim(x_lims)
		plt.ylim(z_lims)


# replace , with . as decimal separator
def conv(x):
    return x.replace(',', '.').encode()


def moving_wind(dat, win_dims):
	return scipy.signal.convolve(dat, np.ones(win_dims))/win_dims


def iterpolate_horizon(x, z):
	xnew = np.linspace(np.min(x), np.max(x), (int)((np.max(x) - np.min(x))/ddz))
	f2 = interp1d(x, z, kind='cubic')
	return xnew.astype(int), (f2(xnew)).astype(int)

def plot_intervals(data, data_path):
	data_3= np.sum(np.abs(data[:, :]), axis=0)/data.shape[0]
	data_check = np.zeros(data.shape)
	
	
	plt.figure(figsize=[30, 15])
	
	for i in range(data.shape[0]):
		data_check[i, :] =  moving_wind(np.abs(data[i, :]), 100)[:-99]
		plt.plot(data_check[i, :], color='k', linewidth=0.5)
		
	plt.plot(moving_wind(data_3, 100), color='r', linewidth=5)
	
	plt.title('Check trand stability')
	plt.grid(True)
	
	plt.ylim([0, 3.5*1e6])
	plt.savefig(data_path + 'stability_' + datafiles[k][:-4], dpi=300)
	plt.clf()
	
	plt.figure(figsize=[30, 15])
	
	plt.errorbar( np.linspace(0, data_3.shape[0]-1, data_3.shape[0]), moving_wind(data_3, 100)[:-99], yerr=np.std(data_check, axis=0), fmt='o', markersize= 2, color='k', ecolor='k', elinewidth=0.5, capsize=2)
	
	plt.title('Check trand stability')
	plt.grid(True)
	
	plt.ylim([0, 3.5*1e6])
	
	plt.savefig(data_path + 'std_' + datafiles[k][:-4], dpi=300)
	plt.clf()

# path with data
data_path = 'data/'

# find the .sgy file in the folder data_path
datafiles = [f[:] for f in os.listdir(data_path) if f[-4:] == '.sgy']

print(datafiles)

	
k = 0
data = read_segy(data_path + datafiles[k])[:60, :]
data_number = datafiles[k][:4]

x_leng = data.shape[0]

bdr_name = 'bedrock ' + data_number + '.txt'
bdr = np.genfromtxt((conv(x) for x in open(data_path + bdr_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]

hsz_name = 'HSZ ' + data_number+'.txt'
HSZ = np.genfromtxt((conv(x) for x in open(data_path + hsz_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]


gain_name = 'gain function ' + data_number+'.txt'
gain_value = np.genfromtxt((conv(x) for x in open(data_path + gain_name)), delimiter=' ', skip_header = 3,  usecols=(0, 2))
	
data_2 = np.sum(np.abs(data[:, :]), axis=0)/data.shape[0]
data_check = np.zeros(data.shape)
	
fig, ax = plt.subplots()
ax.imshow(data.T, aspect='auto', vmin = -1e6, vmax = 1e6)
ax.plot(HSZ/0.448)
ax.plot(bdr/0.448)

# Create a Rectangle patch
rect = patches.Rectangle((0, np.mean(HSZ/0.448)), x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448), linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
plt.savefig('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/old_' + str(data_number) + '/' + 'final_cut', dpi=300)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/old_' + str(data_number) + '/' + 'data_cut', data)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/old_' + str(data_number) + '/' + 'area_interest', np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
plt.clf()

data_2 = np.zeros(data.shape)
data_2[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] = data[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] 
plot_intervals(data_2, '/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/old_' + str(data_number) + '/')

k = 0
data = read_segy(data_path + datafiles[k])[60:135, :]
data_number = datafiles[k][:4]

x_leng = data.shape[0]

bdr_name = 'bedrock ' + data_number + '.txt'
bdr = np.genfromtxt((conv(x) for x in open(data_path + bdr_name)), delimiter='\t', skip_header = 1,  usecols=(5))[60:60+x_leng]

hsz_name = 'HSZ ' + data_number+'.txt'
HSZ = np.genfromtxt((conv(x) for x in open(data_path + hsz_name)), delimiter='\t', skip_header = 1,  usecols=(5))[60:60+x_leng]


gain_name = 'gain function ' + data_number+'.txt'
gain_value = np.genfromtxt((conv(x) for x in open(data_path + gain_name)), delimiter=' ', skip_header = 3,  usecols=(0, 2))
	
data_2 = np.sum(np.abs(data[:, :]), axis=0)/data.shape[0]
data_check = np.zeros(data.shape)
	
fig, ax = plt.subplots()
ax.imshow(data.T, aspect='auto', vmin = -1e6, vmax = 1e6)
ax.plot(HSZ/0.448)
ax.plot(bdr/0.448)

# Create a Rectangle patch
rect = patches.Rectangle((0, np.mean(HSZ/0.448)), x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448), linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
plt.savefig('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/' + 'final_cut', dpi=300)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/' + 'data_cut', data)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/' + 'area_interest', np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
	
print(np.array([[0, (np.mean(HSZ)/0.448)/34], [x_leng - 1, (np.mean(bdr)/0.448)/34]]))
	
plt.clf()

data_2 = np.zeros(data.shape)
data_2[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] = data[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] 
plot_intervals(data_2, '/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/')

k = 1
data = read_segy(data_path + datafiles[k])[:110, :]
data_number = datafiles[k][:4]

x_leng = data.shape[0]

bdr_name = 'bedrock ' + data_number + '.txt'
bdr = np.genfromtxt((conv(x) for x in open(data_path + bdr_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]

hsz_name = 'HSZ ' + data_number+'.txt'
HSZ = np.genfromtxt((conv(x) for x in open(data_path + hsz_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]


gain_name = 'gain function ' + data_number+'.txt'
gain_value = np.genfromtxt((conv(x) for x in open(data_path + gain_name)), delimiter=' ', skip_header = 3,  usecols=(0, 2))
	
data_2 = np.sum(np.abs(data[:, :]), axis=0)/data.shape[0]
data_check = np.zeros(data.shape)
	
fig, ax = plt.subplots()
ax.imshow(data.T, aspect='auto', vmin = -1e6, vmax = 1e6)
ax.plot(HSZ/0.448)
ax.plot(bdr/0.448)

# Create a Rectangle patch
rect = patches.Rectangle((0, np.mean(HSZ/0.448)), x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448), linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
plt.savefig('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/final_cut', dpi=300)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/data_cut', data)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number)+ '/area_interest', np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
	
print(np.array([[0, (np.mean(HSZ)/0.448)/34], [x_leng - 1, (np.mean(bdr)/0.448)/34]]))
plt.clf()


data_2 = np.zeros(data.shape)
data_2[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] = data[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] 
plot_intervals(data_2, '/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/')

	
k = 2
data = read_segy(data_path + datafiles[k])[:111, :]
data_number = datafiles[k][:4]

x_leng = data.shape[0]

bdr_name = 'bedrock ' + data_number + '.txt'
bdr = np.genfromtxt((conv(x) for x in open(data_path + bdr_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]

hsz_name = 'HSZ ' + data_number+'.txt'
HSZ = np.genfromtxt((conv(x) for x in open(data_path + hsz_name)), delimiter='\t', skip_header = 1,  usecols=(5))[:x_leng]


gain_name = 'gain function ' + data_number+'.txt'
gain_value = np.genfromtxt((conv(x) for x in open(data_path + gain_name)), delimiter=' ', skip_header = 3,  usecols=(0, 2))
	
data_2 = np.sum(np.abs(data[:, :]), axis=0)/data.shape[0]
data_check = np.zeros(data.shape)
	
fig, ax = plt.subplots()
ax.imshow(data.T, aspect='auto', vmin = -1e6, vmax = 1e6)
ax.plot(HSZ/0.448)
ax.plot(bdr/0.448)

# Create a Rectangle patch
rect = patches.Rectangle((0, np.mean(HSZ/0.448)), x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448), linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)
plt.savefig('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/final_cut', dpi=300)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/data_cut', data)
np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number)+ '/area_interest', np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
plt.clf()

print(np.array([[0, (np.mean(HSZ)/0.448)/34], [x_leng - 1, (np.mean(bdr)/0.448)/34]]))

data_2 = np.zeros(data.shape)
data_2[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] = data[:, int(np.mean(HSZ/0.448)): int(np.mean(bdr/0.448))] 
plot_intervals(data_2, '/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/')

