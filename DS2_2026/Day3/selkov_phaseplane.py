import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def selkov_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]
    
    [ a , b ] = param


    v1 = b
    v2 = a * y + x**2 * y
    v3 = x
    
    dxdt = v2 - v3
    dydt = v1 - v2
    
    return [ dxdt , dydt ] 


init = [ 1.0 , 1.0 ]
t_span = [ 0 , 100 ]
t_eval = np.linspace( t_span[0] , t_span[1] , 10000 )

a = 0.06
b = 0.6

param = [ a , b ]

solution = solve_ivp( selkov_model, t_span, init, t_eval=t_eval, method='RK45', args=param )

ADP = solution.y[0]
F6P = solution.y[1]

plt.plot( ADP , F6P , 'k-' , label='trajectory' )

x = np.linspace( 0, 3, 100 )
x_null = x / ( a + x**2 ) # The nullcline for dxdt=0
y_null = b / ( a + x**2 ) # The nullcline for dydt=0

plt.plot( x , x_null , 'r-' , label='dx/dt = 0' )
plt.plot( x , y_null , 'b-' , label='dy/dt = 0' )
plt.plot( b , b / ( a + b**2 ) , 'ko' , fillstyle='none' , label='fixed point' )

plt.xlim( 0 , 3 )
plt.ylim( 0 , 4 )
plt.xlabel( 'x (ADP)' )
plt.ylabel( 'y (F6P)' )
plt.legend( loc='upper right' )
plt.savefig( 'selkov_phaseplane.png' , dpi=150 )
plt.show()
