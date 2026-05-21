import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# creating figuresize
fig = plt.figure(figsize=(10,8))

# generating a 3d sine wave
ax = plt.axes(projection='3d')

# creating axes
axes = [5,5,5]

# create data
data = np.ones(axes)

# control transparency
alpha = 0.9

# Control colour
colors = np.empty(axes + [4])

colors[0] = [1, 0, 0, alpha]  # red
colors[1] = [0, 1, 0, alpha]  # green
colors[2] = [0, 0, 1, alpha]  # blue
colors[3] = [1, 1, 0, alpha]  # yellow
colors[4] = [1, 1, 1, alpha]  # grey

# Creating array points using numpy
x = np.arange(0, 20, 0.1)
y = np.cos(x)
z = y*np.sin(x)
c = x + y

# To create a scatter graph
# ax.scatter(x, y, z, c=c)

# turn off/on axis
plt.axis('off')

# Voxels is used to customizations of  the sizes, positions and colors.
ax.voxels(data, facecolors=colors, edgecolors='grey')
plt.show()