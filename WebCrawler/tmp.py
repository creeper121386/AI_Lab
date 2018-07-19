import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

f = plt.figure()
x = np.linspace(-1, 1, 50)
'''x,y = np.meshgrid(x,y)
y = np.linspace(-1, 1, 50)
ax = Axes3D(f)

z = x**2+y**2
ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap=plt.cm.rainbow, edgecolors='black')'''



plt.subplot(2,2,1)
plt.plot(x, x**2, linewidth=1.5, color='red')
plt.subplot(2,2,3)
plt.plot(x, x**2, linewidth=1.5, color='red')
plt.show()
