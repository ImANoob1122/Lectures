import matplotlib.pyplot as plt
import numpy as np

N = 10
x = np.linspace( 0, 3, N )
y = np.linspace( 0, 10, N )

[X, Y] = np.meshgrid( x, y )

a = 0.06
b = 0.6

v1 = b
v2 = a * Y + X**2 * Y
v3 = X

dxdt = v2 - v3
dydt = v1 - v2

#plt.quiver( X, Y, dxdt, dydt )
plt.streamplot( X, Y, dxdt, dydt )
plt.show()
