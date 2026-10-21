import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ALPHA = 0.1
BETA = 10
RHO = 25

def ozbudak_hill_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , beta , rho , x_out , n ] = param

    dxdt = -x + alpha * x_out + beta * y
    dydt = x**n / ( rho + x**n ) - y

    return [ dxdt , dydt ]

def hill_dose_response( ax , x_out , y_init , n , marker ):
    init = [ 0 , y_init ]
    t_span = [ 0 , 60 ]
    param = [ ALPHA , BETA , RHO , x_out , n ]

    solution = solve_ivp( ozbudak_hill_model, t_span, init , method='RK45', args=param )

    y_end = solution.y[1][ -1 ]
    ax.plot( x_out , y_end , marker , fillstyle='none')

    return y_end

def hill_sweep( ax , n ):
    y_init = 0
    for x_out in range(11):
        y_init = hill_dose_response( ax , x_out , y_init , n , 'ko' )
    for x_out in reversed( range(11) ):
        y_init = hill_dose_response( ax , x_out , y_init , n , 'r+' )

    ax.set_title( f'n = {n}' )
    ax.set_xlabel( 'X_out (TMG_out)' )
    ax.set_ylabel( 'y_end (LacY-GFP)' )

# 定常状態は dx/dt = 0 から y = (x - alpha*x_out)/beta を dy/dt = 0 に代入した
# x^(n+1) - (alpha*x_out + beta) x^n + rho x - rho alpha x_out = 0 の正の実根
def count_steady_states( x_out , n ):
    a = ALPHA * x_out
    coef = np.zeros( n + 2 )
    coef[ 0 ] = 1
    coef[ 1 ] = -( a + BETA )
    coef[ n ] += RHO
    coef[ n + 1 ] += -RHO * a
    roots = np.roots( coef )
    real = roots[ np.abs( roots.imag ) < 1e-9 ].real
    return np.sum( real >= a - 1e-12 )

def bistable_range( n , x_out_max=10 ):
    x_outs = np.linspace( 0 , x_out_max , 10001 )
    bistable = [ x for x in x_outs if count_steady_states( x , n ) >= 3 ]
    if not bistable:
        return None
    return ( min( bistable ) , max( bistable ) )

##
## Main
##

fig , ax = plt.subplots()
hill_sweep( ax , 1 )
fig.savefig( 'hill1_dose_response.png' , dpi=150 )

fig , axes = plt.subplots( 2 , 3 , figsize=( 12 , 7 ) )
for n , ax in zip( range( 1 , 6 ) , axes.flat ):
    hill_sweep( ax , n )
axes.flat[ -1 ].axis( 'off' )
fig.tight_layout()
fig.savefig( 'hill_n1to5_dose_response.png' , dpi=150 )

for n in range( 1 , 6 ):
    r = bistable_range( n )
    text = 'none' if r is None else f'{r[0]:.2f} - {r[1]:.2f}'
    print( f'n = {n}: bistable X_out range in [0, 10] = {text}' )

plt.show()
