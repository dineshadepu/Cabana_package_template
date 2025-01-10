"""
"""
import shutil
import sys
import os
import numpy as np
import subprocess

output_dir = sys.argv[1]

shutil.copy('./build/examples/TestComparePython02SerialLoop', output_dir)

os.system('cd ' + output_dir + '&& ./TestComparePython02SerialLoop')


# create the mesh
x, y = np.mgrid[-5.:6., -5.:6.]


x = x.ravel()
y = y.ravel()

# total = np.zeros_like(x)


# neigh_radius = 1.1
# for i in range(len(x)):
#     for j in range(len(x)):
#         if i != j:
#             dx = x[i] - x[j]
#             dy = y[i] - y[j]
#             dist = (dx**2. + dy**2.)**0.5
#             if dist < neigh_radius:
#                 total[i] += 1


# # print(len(total))

# total.resize(11, 11)
# # print(total)


# print("x before is ")
# print(x)
for i in range(len(x)):
    x[i] += 0.5

# print(x)

path = sys.argv[1]

res_npz = os.path.join(path, "results.npz")
np.savez(res_npz,
         x=x,
         y=y)
