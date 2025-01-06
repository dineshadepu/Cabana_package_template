import numpy as np
# create the mesh
x, y = np.mgrid[-5.:6., -5.:6.]


x = x.ravel()
y = y.ravel()

sum_ = np.zeros_like(x)


neigh_radius = 1.1
for i in range(len(x)):
    for j in range(len(x)):
        if i != j:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = (dx**2. + dy**2.)**0.5
            if dist < neigh_radius:
                sum_[i] += 1


# print(len(sum_))

sum_.resize(11, 11)
print(sum_)


print("x before is ")
print(x)
for i in range(len(x)):
    x[i] += 0.5
print("x after moving is ")
print(x)
