import matplotlib.pyplot as plt
import numpy as np

k = 0.2
glc0 = 1.0

t = np.linspace( 0 , 5.0 , 100 )
glc = glc0 * np.exp( - k * t )

plt.plot( t , glc )
plt.xlabel( "t [sec]" )
plt.ylabel( "[Glucose] [mM]" )
plt.savefig( "glucose_analytic.png" )
