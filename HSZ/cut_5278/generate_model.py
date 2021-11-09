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

string_2 = "lamb = " + str(lamb) + "\nrs = " + str(rs) + "\ndims = " + str(dims) + "\nice_z = " + str(HSZ_z) + "\nHSZ_spess = " + str(HSZ_spess) + "\ndomain_x = " + str(domain_x)

string_3 = """
from gprMax.input_cmd_funcs import *
dxx = 0.01
if rs== 0:
        for i in range(0, (int)(lamb*((domain_x*HSZ_spess)/(dxx**2))/(dims**2))):
                idx_x = np.random.randint(0, domain_x//dxx - dims)
                idx_z = np.random.randint(ice_z//dxx, (HSZ_spess + ice_z)//dxx - dims)
                box(idx_x * dxx, idx_z * dxx, 0, idx_x *dxx + dxx * dims, idx_z * dxx + dxx * dims, 1,'Debris')
else:
        for i in range(0, (int)((lamb*(((20*20)/(dxx**2))/(dims**2))*np.pi)/4)):
                idx_x = np.random.randint(dims//2, 2000 - dims//2)
                idx_z = np.random.randint(1000 + dims//2, 3000 - dims//2)
                cylinder(idx_x * dxx, idx_z * dxx, 0, idx_x *dxx,idx_z * dxx, 1, dims//2,'Debris')
                
#end_python:

--------------------------------

#rx: 2.5 0.1 0
#src_steps: 1.0 0.0 0.0
#rx_steps: 1.0 0.0 0.0
#waveform: gaussiandotnorm 1.0 250e6 Sr250MHz
#hertzian_dipole: z 2.5 0.1 0 Sr250MHz"""

with open('models/model_' + str(dims) + str(lamb) + str(i) +'.in', 'w+') as f:
  f.write(string)
  f.write(string0)
  f.write(string_2)
  f.write(string_3)
