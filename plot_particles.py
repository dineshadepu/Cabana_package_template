import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "./build/particles_0.h5"
f = h5py.File(file_name, "r")
# print(np.array(f["radius"]))
x = np.array(f["positions"][:, 0])
y = np.array(f["positions"][:, 1])
plt.scatter(x, y)
plt.show()
