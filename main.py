import numpy as np
import hdf5storage

def load_data(year,month,day):
    fiepath = 'data/calibration/'
    filename = f"{year}_{month:02d}_{day:02d}.mat"

    mat = hdf5storage.loadmat(fiepath + filename)

    return mat


mat = load_data(2019, 1, 2)

print(mat.keys())
for key in mat.keys():
    print(key)
    print(np.max(mat[key]))
