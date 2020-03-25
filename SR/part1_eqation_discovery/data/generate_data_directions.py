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

data = load_hdf5('..\TTK4853-EiT-Hybrid-modellering-og-analyse\SR\data\\validation\\2019_05_05.mat')
#Keys: ['air_potential_temperature_ml', 'air_pressure_ml', 'turbulence_dissipation_ml',
#       'turbulence_index_ml', 'upward_air_velocity_ml', 'x_wind_ml', 'y_wind_ml']

h_start = 0 #Houre
h_end = 1
i_min = 0; i_max = 120
j_min = 0; j_max = 120
k_min = 0; k_max = 39 

#geo = pd.DataFrame(columns=['geo'])
#for i in range(100):
#    for j in range(100):
#        temp = pd.DataFrame([[\
#            data['geopotential_height_ml'][0,31,i,j]
#            ]],columns=['geo'])
#        geo = geo.append(temp)
#geo.to_csv('geo', index=False)

names = list(['u', 'ux','uy','uz','u2x','u2y','u2z','uux','vuy','wuz','p','px',])


x = np.linspace(0,120, 120)
print(x)
y = np.linspace(0,120, 120)
X, Y = np.meshgrid(x, y)
Z1 = np.zeros((120,120))
print(data['geopotential_height_ml'][0,:,1,1])
for i in range(120):
    for j in range(120):
        Z1[i,j] = data['geopotential_height_ml'][0,39,j,i]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z1, 50, cmap='binary')
#plt.show()
Z3 = np.zeros((120,120))
for i in range(120):
    for j in range(120):
        Z3[i,j] = data['x_wind_ml'][0,39,j,i]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z3, 50, cmap='binary')

#names = list(['u', 'ux','uy','u2x','u2y','p','px',])
for h in range(h_start, h_end):
    print('h = ', h)
    for x in range(i_min, i_max):
        for y in range(j_min, j_max):
            data['x_wind_ml'][h,k_min:k_max,y,x], data['y_wind_ml'][h,k_min:k_max,y,x], data['upward_air_velocity_ml'][h,k_min:k_max,y,x], data['air_pressure_ml'][h,k_min:k_max,y,x] \
                = interpolation(k_max-k_min,data['geopotential_height_ml'][h,k_min:k_max,y,x],\
                    data['x_wind_ml'][h,k_min:k_max,y,x],\
                    data['y_wind_ml'][h,k_min:k_max,y,x],\
                    data['upward_air_velocity_ml'][h,k_min:k_max,y,x],\
                    data['air_pressure_ml'][h,k_min:k_max,y,x])

Z2 = np.zeros((120,120))
for i in range(120):
    for j in range(120):
        Z2[i,j] = data['air_pressure_ml'][0,2,j,i]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z2, 50, cmap='binary')
plt.show()

derivatives = pd.DataFrame(columns=names)
for h in range(h_start, h_end):
    print('h = ',h)
    for k in range(k_min,k_max):
        print('k = ',k)
        for x in range(i_min,i_max):
            for y in range(j_min,j_max):
                current_derivatives = pd.DataFrame([[data['x_wind_ml'][h,k,y,x], \
                    ux(data['x_wind_ml'][h,k,y,x+1],data['x_wind_ml'][h,k,y,x-1]),\
                    uy(data['x_wind_ml'][h,k,y+1,x],data['x_wind_ml'][h,k,y-1,x]),\
                    uz(data['x_wind_ml'][h,k+1,y,x],data['x_wind_ml'][h,k-1,y,x]),\
                    u2x(data['x_wind_ml'][h,k,y,x+1],data['x_wind_ml'][h,k,y,x],data['x_wind_ml'][h,k,y,x-1]),\
                    u2y(data['x_wind_ml'][h,k,y+1,x],data['x_wind_ml'][h,k,y,x],data['x_wind_ml'][h,k,y-1,x]),\
                    u2z(data['x_wind_ml'][h,k+1,y,x],data['x_wind_ml'][h,k,y,x],data['x_wind_ml'][h,k-1,y,x]),\
                    data['x_wind_ml'][h,k,y,x]*ux(data['x_wind_ml'][h,k,y,x+1],data['x_wind_ml'][h,k,y,x-1]),\
                    data['y_wind_ml'][h,k,y,x]*uy(data['x_wind_ml'][h,k,y+1,x],data['x_wind_ml'][h,k,y-1,x]),\
                    data['upward_air_velocity_ml'][h,k,y,x]*uz(data['x_wind_ml'][h,k+1,y,x],data['x_wind_ml'][h,k-1,y,x]),\
                    data['air_pressure_ml'][h,k,y,x],\
                    px(data['air_pressure_ml'][h,k,y,x+1],data['air_pressure_ml'][h,k,y,x-1])\
                    ]],columns=names)
                derivatives = derivatives.append(current_derivatives)


derivatives.to_csv('navier_stokes_data_u.csv',index=False)
print("X done")
derivatives = pd.DataFrame(columns=names)
names = list(['v','vx','vy','vz','v2x','v2y','v2z','p','py',])
for i in range(i_min,i_max):
    for j in range(j_min,j_max):
        for k in range(k_min,k_max):
            current_derivatives = pd.DataFrame([[\
                data['y_wind_ml'][i,j,k,h], \
                vx(data['y_wind_ml'][i+1,j,k,h],data['y_wind_ml'][i-1,j,k,h]),\
                vy(data['y_wind_ml'][i,j+1,k,h],data['y_wind_ml'][i,j-1,k,h]),\
                vz(data['y_wind_ml'][i,j,k+1,h],data['y_wind_ml'][i,j,k-1,h]),\
                v2x(data['y_wind_ml'][i+1,j,k,h],data['y_wind_ml'][i,j,k,h],data['y_wind_ml'][i-1,j,k,h]),\
                v2y(data['y_wind_ml'][i,j+1,k,h],data['y_wind_ml'][i,j,k,h],data['y_wind_ml'][i,j-1,k,h]),\
                v2z(data['y_wind_ml'][i,j,k+1,h],data['y_wind_ml'][i,j,k,h],data['y_wind_ml'][i,j,k-1,h]),\
                data['air_pressure_ml'][i,j,k,h],\
                py(data['air_pressure_ml'][i,j+1,k,h],data['air_pressure_ml'][i,j-1,k,h]),\
                ]], columns=names)
            derivatives = derivatives.append(current_derivatives)
derivatives['vvx'] = derivatives['v']*derivatives['vx']
derivatives['vvy'] = derivatives['v']*derivatives['vy']
derivatives['vvz'] = derivatives['v']*derivatives['vz']

derivatives.to_csv('navier_stokes_data_v.csv',index=False)
print("Y done")
derivatives = pd.DataFrame(columns=names)
names = list(['w','wx','wy','wz','w2x','w2y','w2z','p','pz'])
for i in range(i_min,i_max):
    for j in range(j_min,j_max):
        for k in range(k_min,k_max):
            current_derivatives = pd.DataFrame([[
                data['upward_air_velocity_ml'][i,j,k,h], \
                wx(data['upward_air_velocity_ml'][i+1,j,k,h],data['upward_air_velocity_ml'][i-1,j,k,h]),\
                wy(data['upward_air_velocity_ml'][i,j+1,k,h],data['upward_air_velocity_ml'][i,j-1,k,h]),\
                wz(data['upward_air_velocity_ml'][i,j,k+1,h],data['upward_air_velocity_ml'][i,j,k-1,h]),\
                w2x(data['upward_air_velocity_ml'][i+1,j,k,h],data['upward_air_velocity_ml'][i,j,k,h],data['upward_air_velocity_ml'][i-1,j,k,h]),\
                w2y(data['upward_air_velocity_ml'][i,j+1,k,h],data['upward_air_velocity_ml'][i,j,k,h],data['upward_air_velocity_ml'][i,j-1,k,h]),\
                w2z(data['upward_air_velocity_ml'][i,j,k+1,h],data['upward_air_velocity_ml'][i,j,k,h],data['upward_air_velocity_ml'][i,j,k-1,h]),\
                data['air_pressure_ml'][i,j,k,h],\
                pz(data['air_pressure_ml'][i,j,k+1,h],data['air_pressure_ml'][i,j,k-1,h])]], columns=names)
            derivatives = derivatives.append(current_derivatives)

derivatives['wwx'] = derivatives['w']*derivatives['wx']
derivatives['wwy'] = derivatives['w']*derivatives['wy']
derivatives['wwz'] = derivatives['w']*derivatives['wz']
print("Z done")


derivatives.to_csv('navier_stokes_data_w.csv',index=False)
