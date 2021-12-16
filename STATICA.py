import numpy as np
import os 

data = np.genfromtxt('data/5728_tab_csv.csv', delimiter=',', skip_header= 1)
#print(data.shape)

tab = np.zeros((data.shape[0],7))
#print(tab.shape)

vel = 0.17
length_profile = 305

tab[:,:3] = data[:,:]
tab[:,3] = np.max(tab[:,2])
tab[:,4] = tab[:,3] - tab[:,2]
tab[:,5] = np.max(tab[:,4])
tab[:,6] = tab[:,5] - tab[:,4]
X = tab[:,0]
Y = tab[:,1]
STAT = tab[:,6]

print('MAX Z: ', tab[0,2], 'm')
print('MAX STAT [m]: ', np.max(STAT))
print('lunghezza traccia: ', (length_profile + (np.max(STAT) *2)/vel), 'ns')
 
np.savetxt('XYZ/5728_X.txt', X, fmt='%1.2f' )
np.savetxt('XYZ/5728_Y.txt', Y, fmt='%1.2f')
np.savetxt('XYZ/5728_Z_STAT.txt', STAT, fmt='%1.2f')
