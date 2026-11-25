import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

a = 0.06
b = 0.6
fix = np.array( [ b , b / ( a + b**2 ) ] )


def selkov_model( t , init ):
    x , y = init
    v1 = b
    v2 = a * y + x**2 * y
    v3 = x
    return [ v2 - v3 , v1 - v2 ]


init = fix + np.array( [ 0.01 , 0.0 ] )
t_eval = np.linspace( 0 , 60 , 6000 )
sol = solve_ivp( selkov_model , [0 , 60] , init , t_eval=t_eval , method='RK45' , rtol=1e-9 , atol=1e-12 )

fig , (ax1 , ax2) = plt.subplots( 1 , 2 , figsize=(10, 4) )
ax1.plot( sol.t , sol.y[0] , 'k-' , label='ADP' )
ax1.plot( sol.t , sol.y[1] , 'b-' , label='F6P' )
ax1.set_xlabel( 'time' )
ax1.set_ylabel( 'concentration' )
ax1.legend( loc='upper left' )
ax2.plot( sol.y[0] , sol.y[1] , 'k-' )
ax2.plot( fix[0] , fix[1] , 'ko' , fillstyle='none' )
ax2.set_xlabel( 'x (ADP)' )
ax2.set_ylabel( 'y (F6P)' )
fig.tight_layout()
fig.savefig( 'selkov_perturbation.png' , dpi=150 )
plt.show()
