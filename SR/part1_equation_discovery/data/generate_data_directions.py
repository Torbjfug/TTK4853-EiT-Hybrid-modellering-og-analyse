import h5py
import os
import numpy as np
import hdf5storage
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Interpolering import interpolation
import platform
#dx = 221
dx = 221
dy = 221
dz = 55


def save_hdf5(data, filename):
    with h5py.File(filename, 'w') as data_file:
        for key in data.keys():
            data_file.create_dataset(key, data=data[key],)


def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in list(f.keys()):
            data[key] = np.array(f[key])
    return data


def concatinate_files(data1, data2):
    for key in data2.keys():
        data1[key] = np.concatenate((data1[key], data2[key]))
    return data1


def concatenate_hdf5(filename, data):
    with h5py.File(filename, 'a') as f:
        dataset_keys = list(f.keys())
        for key in data.keys():
            if key in dataset_keys:
                dset = f[key]
                dset.resize(dset.shape[0]+data[key].shape[0], axis=0)
                dset[-data[key].shape[0]:] = data[key]
            else:
                max_size = (None,)+data[key].shape[1:]
                f.create_dataset(key, data=data[key], maxshape=max_size)
    return data


def concatenate_files_in_folder(source_dir, save_file):
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.mat'):
            data = hdf5storage.loadmat(source_dir + filename)
            concatenate_hdf5(save_file, data)
    all_data = load_hdf5(save_file)
    return all_data


ux = lambda ui_pluss, ui_minus : (ui_pluss-ui_minus)/2/dx
u2x = lambda ui_pluss, ui, ui_minus : (ui_pluss-2*ui+ui_minus)/dx**2
uy = lambda uj_pluss, uj_minus : (uj_pluss-uj_minus)/2/dy
u2y = lambda uj_pluss, uj, uj_minus : (uj_pluss-2*uj+uj_minus)/dy**2
uz = lambda uk_pluss, uk_minus : (uk_pluss-uk_minus)/2/dz
u2z = lambda uk_pluss, uk, uk_minus : (uk_pluss-2*uk+uk_minus)/dz**2
ut = lambda u_pluss, u_now : (u_pluss-u_now)/3600

vx = lambda vi_pluss, vi_minus : (vi_pluss-vi_minus)/2/dx
v2x = lambda vi_pluss, vi, vi_minus : (vi_pluss-2*vi+vi_minus)/dx**2
vy = lambda vj_pluss, vj_minus : (vj_pluss-vj_minus)/2/dy
v2y = lambda vj_pluss, vj, vj_minus : (vj_pluss-2*vj+vj_minus)/dy**2
vz = lambda vk_pluss, vk_minus : (vk_pluss-vk_minus)/2/dz
v2z = lambda vk_pluss, vk, vk_minus : (vk_pluss-2*vk+vk_minus)/dz**2

wx = lambda wi_pluss, wi_minus : (wi_pluss-wi_minus)/2/dx
w2x = lambda wi_pluss, wi, wi_minus : (wi_pluss-2*wi+wi_minus)/dx**2
wy = lambda wj_pluss, wj_minus : (wj_pluss-wj_minus)/2/dy
w2y = lambda wj_pluss, wj, wj_minus : (wj_pluss-2*wj+wj_minus)/dy**2
wz = lambda wk_pluss, wk_minus : (wk_pluss-wk_minus)/2/dz
w2z = lambda wk_pluss, wk, wk_minus : (wk_pluss-2*wk+wk_minus)/dz**2

px = lambda pi_pluss, pi_minus : (pi_pluss-pi_minus)/2/dx
py = lambda pj_pluss, pj_minus : (pj_pluss-pj_minus)/2/dy
pz = lambda pk_pluss, pk_minus : (pk_pluss-pk_minus)/2/dz


# For windows:
#data = load_hdf5('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\data\\validation\\2019_05_05.mat')
# For linux:
if(platform.system() == 'Windows'): #Windows
    data = load_hdf5('..\TTK4853-Eit-Hybrid-modellering-og-analyse\SR\data\\validation/2019_05_05.mat')
else:
    data = load_hdf5('/home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/data/validation/2019_05_05.mat')
#Keys: ['air_potential_temperature_ml', 'air_pressure_ml', 'turbulence_dissipation_ml',
#       'turbulence_index_ml', 'upward_air_velocity_ml', 'x_wind_ml', 'y_wind_ml']

h_start = 0 #Houre
h_end = 11
i_min = 0; i_max = 133
j_min = 0; j_max = 133
k_min = 0; k_max = 40 

#geo = pd.DataFrame(columns=['geo'])
#for i in range(100):
#    for j in range(100):
#        temp = pd.DataFrame([[\
#            data['geopotential_height_ml'][0,31,i,j]
#            ]],columns=['geo'])
#        geo = geo.append(temp)
#geo.to_csv('geo', index=False)

names = list(['u', 'ux','uy','uz','u2x','u2y','u2z','uux','vuy','wuz','p','px',])

for h in range(h_start, h_end):
    print('h = ', h)
    for x in range(i_min, i_max):
        for y in range(j_min, j_max):
            data['x_wind_ml'][h,k_min:k_max,y,x],\
                data['y_wind_ml'][h,k_min:k_max,y,x],\
                data['upward_air_velocity_ml'][h,k_min:k_max,y,x],\
                data['air_pressure_ml'][h,k_min:k_max,y,x]\
                = interpolation(k_max-k_min,data['geopotential_height_ml'][h,k_min:k_max,y,x],\
                    data['x_wind_ml'][h,k_min:k_max,y,x],\
                    data['y_wind_ml'][h,k_min:k_max,y,x],\
                    data['upward_air_velocity_ml'][h,k_min:k_max,y,x],\
                    data['air_pressure_ml'][h,k_min:k_max,y,x])


i_min = 20; i_max = 100
j_min = 20; j_max = 100
k_min = 0; k_max = 40 

i_min += 1; i_max -= 1
j_min += 1; j_max -= 1
k_min += 1; k_max -= 1

_ux = ux(data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])


Z2 = np.zeros((48,48))

x = np.linspace(0,48, 48)
y = np.linspace(0,48, 48)
X, Y = np.meshgrid(x, y)

#for i in range(48):
#    for j in range(48):
#        Z2[i,j] = data['x_wind_ml'][0,20,j,i]
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z2, 50, cmap='binary')

_uy = ux(data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])


_uz = ux(data['x_wind_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_u2x = u2x(data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

for i in range(48):
    for j in range(48):
        Z2[i,j] = _u2x[0,20,j,i]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z2, 50, cmap='binary')   
plt.show()


_u2y = u2x(data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_u2z = u2x(data['x_wind_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['x_wind_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_uux = data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_ux

_vuy = data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_uy

_wuz = data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_uz

_px = px(data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

_ux = pd.DataFrame(np.reshape(_ux,(-1,1)), columns=list(['ux']))
_uy = pd.DataFrame(np.reshape(_uy, (-1,1)), columns=list(['uy']))
_uz = pd.DataFrame(np.reshape(_uz, (-1,1)), columns=list(['uz']))
_u2x = pd.DataFrame(np.reshape(_u2x, (-1,1)), columns=list(['u2x']))
_u2y = pd.DataFrame(np.reshape(_u2y, (-1,1)), columns=list(['u2y']))
_u2z = pd.DataFrame(np.reshape(_u2z, (-1,1)), columns=list(['u2z']))
_uux = pd.DataFrame(np.reshape(_uux, (-1,1)), columns=list(['uux']))
_vuy = pd.DataFrame(np.reshape(_vuy, (-1,1)), columns=list(['vuy']))
_wuz = pd.DataFrame(np.reshape(_wuz, (-1,1)), columns=list(['wuz']))
_px = pd.DataFrame(np.reshape(_px,(-1,1)), columns=list(['px']))


derivatives = pd.concat([_u2x, _u2y, _u2z, _uux, _vuy, _wuz, _px], axis=1)
if(platform.system() == 'Windows'): #Windows
    derivatives.to_csv('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\part1_equation_discovery\data\\navier_stokes_data_u.csv',index=False)
else: 
    derivatives.to_csv('/home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/part1_equation_discovery/data/navier_stokes_data_u.csv',index=False)
print("X done")


#############################################################################
#############################################################################

_vx = vx(data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

_vy = vx(data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_vz = vx(data['y_wind_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_v2x = v2x(data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

_v2y = v2x(data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_v2z = v2x(data['y_wind_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['y_wind_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_uvx = data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_vx

_vvy = data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_vy

_wvz = data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_vz

_py = py(data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_vx = pd.DataFrame(np.reshape(_vx,(-1,1)), columns=list(['vx']))
_vy = pd.DataFrame(np.reshape(_vy, (-1,1)), columns=list(['vy']))
_vz = pd.DataFrame(np.reshape(_vz, (-1,1)), columns=list(['vz']))
_v2x = pd.DataFrame(np.reshape(_v2x, (-1,1)), columns=list(['v2x']))
_v2y = pd.DataFrame(np.reshape(_v2y, (-1,1)), columns=list(['v2y']))
_v2z = pd.DataFrame(np.reshape(_v2z, (-1,1)), columns=list(['v2z']))
_uvx = pd.DataFrame(np.reshape(_uvx, (-1,1)), columns=list(['uvx']))
_vvy = pd.DataFrame(np.reshape(_vvy, (-1,1)), columns=list(['vvy']))
_wvz = pd.DataFrame(np.reshape(_wvz, (-1,1)), columns=list(['wvz']))

#_p = pd.DataFrame(_ux, columns=list('p'))
_py = pd.DataFrame(np.reshape(_py,(-1,1)), columns=list(['py']))

derivatives = pd.concat([_vx, _vy, _vz, _v2x, _v2y, _v2z, _uvx, _vvy, _wvz, _py], axis=1)
if(platform.system() == 'Windows'): #Windows
    derivatives.to_csv('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\part1_equation_discovery\data\\navier_stokes_data_v.csv',index=False)
else: 
    derivatives.to_csv('/home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/part1_equation_discovery/data/navier_stokes_data_v.csv',index=False)
print("Y done")

#############################################################################
#############################################################################

_wx = wx(data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

_wy = wx(data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_wz = wx(data['upward_air_velocity_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_w2x = w2x(data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min+1:i_max+1],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min-1:i_max-1])

_w2y = w2x(data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_w2z = w2x(data['upward_air_velocity_ml'][h_start:h_end,k_min+1:k_max+1,j_min:j_max,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max],\
    data['upward_air_velocity_ml'][h_start:h_end,k_min-1:k_max-1,j_min:j_max,i_min:i_max])

_uwx = data['x_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_wx

_vwy = data['y_wind_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_wy

_wwz = data['upward_air_velocity_ml'][h_start:h_end,k_min:k_max,j_min:j_max,i_min:i_max]*_wz

_pz = pz(data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min+1:j_max+1,i_min:i_max],\
    data['air_pressure_ml'][h_start:h_end,k_min:k_max,j_min-1:j_max-1,i_min:i_max])

_wx = pd.DataFrame(np.reshape(_wx,(-1,1)), columns=list(['wx']))
_wy = pd.DataFrame(np.reshape(_wy, (-1,1)), columns=list(['wy']))
_wz = pd.DataFrame(np.reshape(_wz, (-1,1)), columns=list(['wz']))
_w2x = pd.DataFrame(np.reshape(_w2x, (-1,1)), columns=list(['w2x']))
_w2y = pd.DataFrame(np.reshape(_w2y, (-1,1)), columns=list(['w2y']))
_w2z = pd.DataFrame(np.reshape(_w2z, (-1,1)), columns=list(['w2z']))
_uwx = pd.DataFrame(np.reshape(_uwx, (-1,1)), columns=list(['uwx']))
_vwy = pd.DataFrame(np.reshape(_vwy, (-1,1)), columns=list(['vwy']))
_wwz = pd.DataFrame(np.reshape(_wwz, (-1,1)), columns=list(['wvz']))

#_p = pd.DataFrame(_ux, columns=list('p'))
_pz = pd.DataFrame(np.reshape(_pz,(-1,1)), columns=list(['pz']))

derivatives = pd.concat([_wx, _wy, _wz, _w2x, _w2y, _w2z, _uwx, _vwy, _wwz, _pz], axis=1)
if(platform.system() == 'Windows'): #Windows
    derivatives.to_csv('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\part1_equation_discovery\data\\navier_stokes_data_w.csv',index=False)
else: 
    derivatives.to_csv('/home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/part1_equation_discovery/data/navier_stokes_data_w.csv',index=False)
print("W done")