import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def ozbudak_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , beta , rho , x_out ] = param

    dxdt = - x + alpha * x_out + beta * y
    dydt = x**2 / ( rho + x**2 ) - y
    
    return [ dxdt , dydt ] 

def hysteresis_timeseries( x_out_init ):
    init = [ 0 , 0 ]
    t_span = [ 0 , 60 ]

    alpha = 0.1
    beta = 10
    rho = 25
    x_out = x_out_init

    param = [ alpha , beta , rho , x_out ]

    solution = solve_ivp( ozbudak_model, t_span, init , method='RK45', args=param )
    
    t = solution.t
    TMG = solution.y[0]
    LacY = solution.y[1]

    plt.figure()
    plt.plot( t , TMG , color='tab:blue' , label='x (TMG_in)' )
    plt.plot( t , LacY , color='tab:orange' , label='y (LacY-GFP)' )

    ##
    ## Repeat after elimitating TMG_out (x_out)
    ##
    
    x_init = TMG[ -1 ]
    y_init = LacY[ -1 ]
    init = [ x_init , y_init ]
    
    x_out = 0
    param = [ alpha , beta , rho , x_out ]
    
    t_span = [ 60 , 120 ]
    
    solution = solve_ivp( ozbudak_model, t_span, init , method='RK45', args=param )

    t = solution.t
    TMG = solution.y[0]
    LacY = solution.y[1]
    
    plt.plot( t , TMG , color='tab:blue')
    plt.plot( t , LacY , color='tab:orange')
    plt.axvline( 60 , color='gray' , linestyle='--' )
    plt.title( f'X_out = {x_out_init} (t < 60), 0 (t >= 60)' )
    plt.xlabel( 'time' )
    plt.ylabel( 'dimensionless concentration' )
    plt.legend()
    plt.savefig( f'hysteresis_xout{x_out_init}.png' , dpi=150 )

for x_out in [ 5 , 10 ]:
    hysteresis_timeseries( x_out )
plt.show()
