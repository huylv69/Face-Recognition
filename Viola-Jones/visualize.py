from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

inputfile = open("result.txt", "r")
scales = []
minNbs = []
minSizes = []
accurates = []
for line in inputfile:
    params = line[:-1].split(" ")
    scales.append(params[0])
    minNbs.append(params[1])
    minSizes.append(params[2])
    accurates.append(params[3])    

x = np.array(scales, dtype=float)
y = np.array(minNbs, dtype=float)
z = np.array(minSizes, dtype=float)
c = np.array(accurates, dtype=float)

s = ax.scatter(x, y, z, c=c, vmin=0, vmax=1, cmap=plt.cm.coolwarm)
cbar = plt.colorbar(mappable=s, ax=ax)
cbar.set_label('Accurate')
ax.set_xlabel('scales')
ax.set_ylabel('minNbs')
ax.set_zlabel('minSize')
plt.show()