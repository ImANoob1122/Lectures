import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def ozbudak_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , beta , rho , x_out ] = param
    
    dxdt = -x + alpha * x_out + beta * y
    dydt = x**2 / ( rho + x**2 ) - y
    
    return [ dxdt , dydt ] 

def phaseplane( x_out ):
    init = [ 0 , 0 ]
    t_span = [ 0 , 30 ]

    alpha = 0.1
    beta = 10
    rho = 25

    param = [ alpha , beta , rho , x_out ]

    solution = solve_ivp( ozbudak_model, t_span, init , method='RK45', args=param )

    TMG = solution.y[0]
    LacY = solution.y[1]

    plt.figure()
    plt.plot( TMG , LacY , 'k-' , label='trajectory' )
    plt.plot( TMG[-1] , LacY[-1] , 'ko' )

    x = np.linspace( 0, 10, 100 )

    x_null = ( x - alpha * x_out ) / beta # The nullcline for dxdt=0
    y_null = x**2 / ( rho + x**2 ) # The nullcline for dydt=0

    plt.plot( x , x_null , 'r-' , label='dx/dt = 0' )
    plt.plot( x , y_null , 'b-' , label='dy/dt = 0' )

    plt.xlim( 0 , 10 )
    plt.ylim( 0 , 1 )
    plt.title( f'X_out = {x_out}' )
    plt.xlabel( 'x (TMG_in)' )
    plt.ylabel( 'y (LacY-GFP)' )
    plt.legend()
    plt.savefig( f'phaseplane_xout{x_out}.png' , dpi=150 )

for x_out in [ 5 , 10 ]:
    phaseplane( x_out )
plt.show()
