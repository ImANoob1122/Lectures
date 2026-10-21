import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# LacY による TMG 取り込み (beta * y) を除くとポジティブ・フィードバックが切れる
def ozbudak_no_feedback_model( t , init , *param ):
    x = init[ 0 ]
    y = init[ 1 ]

    [ alpha , rho , x_out ] = param

    dxdt = -x + alpha * x_out
    dydt = x**2 / ( rho + x**2 ) - y

    return [ dxdt , dydt ]

def no_feedback_dose_response( x_out , y_init , marker ):
    init = [ 0 , y_init ]
    t_span = [ 0 , 60 ]

    alpha = 0.1
    rho = 25

    param = [ alpha , rho , x_out ]

    solution = solve_ivp( ozbudak_no_feedback_model, t_span, init , method='RK45', args=param )

    LacY = solution.y[1]
    y_end = LacY[ -1 ]

    plt.plot( x_out , y_end , marker , fillstyle='none')

    return y_end

##
## Main
##

y_init = 0
for x_out in range(11):
    y_init = no_feedback_dose_response( x_out , y_init , 'ko' )

for x_out in reversed( range(11) ):
    y_init = no_feedback_dose_response( x_out , y_init , 'r+' )

plt.xlabel( 'X_out (TMG_out)' )
plt.ylabel( 'y_end (GFP)' )
plt.savefig( 'no_feedback_dose_response.png' , dpi=150 )
plt.show()
