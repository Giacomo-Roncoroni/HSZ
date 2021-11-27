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


# replace , with . as decimal separator
def conv(x):
    return x.replace(',', '.').encode()


def moving_wind(dat, win_dims):
	return scipy.signal.convolve(dat, np.ones(win_dims))/win_dims

# path with data
data_path = 'data_no_gain/'

# find the .sgy file in the folder data_path
datafiles = [f[:] for f in os.listdir(data_path) if f[-4:] == '.sgy']

print(len(datafiles))

for k in range(len(datafiles)):
	if k == 0: 
		data = read_segy(data_path + datafiles[k])[60:135, :]
	else: 
		data = read_segy(data_path + datafiles[k])[:, :]
	data_number = datafiles[k][:4]
	x_leng = data.shape[0]
	bdr_name = 'bedrock ' + data_number + '.txt'
	bdr = np.genfromtxt((conv(x) for x in open(data_path + bdr_name)), delimiter='\t', skip_header = 1,  usecols=(5))[60:60+x_leng]
	hsz_name = 'HSZ ' + data_number+'.txt'
	HSZ = np.genfromtxt((conv(x) for x in open(data_path + hsz_name)), delimiter='\t', skip_header = 1,  usecols=(5))[60:60+x_leng]
	#np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/data/' + str(data_number) + '/' + 'gain_val', gain_value)
	np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/' + data_path + str(data_number) + '/' + 'base_trace', np.sum(np.abs(data[:, :]), axis=0)/data.shape[0])
	np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/' + data_path + str(data_number) + '/' + 'data_cut', data)
	np.save('/mnt/c/Users/gronc/WORK/06_gprmax_debris/99_real_data/' + data_path + str(data_number) + '/' + 'area_interest', np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
	print( np.array([[0, np.mean(HSZ/0.448)], [x_leng - 1, np.mean(bdr/0.448) - np.mean(HSZ/0.448)]]))
