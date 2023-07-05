import sys

i = int(sys.argv[1])
domain_x = float(sys.argv[2])
domain_z = float(sys.argv[3])
dxx = float(sys.argv[4])
dims = int(sys.argv[5])
HSZ_z = float(sys.argv[6])
HSZ_spess = float(sys.argv[7])
lamb = float(sys.argv[8])
rs = int(sys.argv[9])
time = float(sys.argv[10])
ldeb = float(sys.argv[11])

string = "#title: Model 1" + "\n#domain: " + str(domain_x) + " " + str(domain_z) + " 1 \n#dx_dy_dz: " + str(dxx) + " " + str(dxx) + " 1.0 \n#time_window: " + str(time) 

string0 ="\n\n--------------------------------\n#material: " + str(ldeb) + " 0.0 1.0 0.0 Debris\n#material: 3.2  0.0 1.0 0.0 Ice_1\n#material: 9 0.0 1.0 0.0 B_rock\n\n --------------------------------\n\n#box: 0.0 0.0 0.0 " + str(domain_x) + " " + str(domain_z) + " 1.0 Ice_1\n#box: 0.0 " + str(HSZ_z + HSZ_spess) +" 0.0 " + str(domain_x) + " " + str(domain_z) + " 1.0 B_rock\n\n#python:\nimport numpy as np\n"

string_2 = "lamb = " + str(lamb) + "\nrs = " + str(rs) + "\ndims = " + str(dims) + "\nice_z = " + str(HSZ_z) + "\nHSZ_spess = " + str(HSZ_spess) + "\ndomain_x = " + str(domain_x)+ " \ndomain_z = " + str(domain_z) + " \ni = " + str(i)

string_3 = """
from gprMax.input_cmd_funcs import *
dxx = 0.01
if rs== 0:
        check = np.zeros((int(domain_z//dxx), int(domain_x//dxx)))
        while np.sum(check)*(dxx**2)<(lamb*(domain_x*HSZ_spess)):
                idx_x = np.random.randint(0, domain_x//dxx - dims)
                idx_z = np.random.randint(ice_z//dxx, (HSZ_spess + ice_z)//dxx - dims)
                box(idx_x * dxx, idx_z * dxx, 0, idx_x *dxx + dxx * dims, idx_z * dxx + dxx * dims, 1,'Debris')
                check[int(idx_z):int(idx_z + dims), int(idx_x):int(idx_x + dims)] = 1
else:
    min_dims = 0.13//dxx
    max_dims = 0.60//dxx
    check = np.zeros((int(domain_z//dxx), int(domain_x//dxx)))
    while np.sum(check)*(dxx**2)<(lamb*(domain_x*HSZ_spess)):
                dims = np.random.randint(min_dims, max_dims)
                idx_x = np.random.randint(0, domain_x//dxx - dims)
                idx_z = np.random.randint(ice_z//dxx, (HSZ_spess + ice_z)//dxx - dims)
                box(idx_x * dxx, idx_z * dxx, 0, idx_x *dxx + dxx * dims, idx_z * dxx + dxx * dims, 1,'Debris')
                check[int(idx_z):int(idx_z + dims), int(idx_x):int(idx_x + dims)] = 1
    #np.save('models/model' + str(dims) + str(lamb) + str(i), check)
#end_python:

--------------------------------
#excitation_file: /m100/home/userexternal/groncor1/gprmax/HSZ/MALA_250MHz
#rx: 5.0 0.1 0
#src_steps: 1.0 0.0 0.0
#rx_steps: 5.0 0.1 0.0
#hertzian_dipole: z 5.0 0.05 0 MALA_250MHz"""

with open('models/model_' + str(dims) + str(lamb) + str(i) +'.in', 'w+') as f:
  f.write(string)
  f.write(string0)
  f.write(string_2)
  f.write(string_3)
