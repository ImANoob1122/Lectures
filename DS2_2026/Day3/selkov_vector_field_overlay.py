import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

a = 0.06
b = 0.6


def selkov_rhs( x , y ):
    v1 = b
    v2 = a * y + x**2 * y
    v3 = x
    return v2 - v3 , v1 - v2


def selkov_model( t , init ):
    return selkov_rhs( init[0] , init[1] )


def plot_vector_fields():
    N = 10
    [X, Y] = np.meshgrid( np.linspace( 0, 3, N ) , np.linspace( 0, 10, N ) )
    dxdt , dydt = selkov_rhs( X , Y )

    fig , (ax1 , ax2) = plt.subplots( 1 , 2 , figsize=(10, 4.2) )
    ax1.quiver( X , Y , dxdt , dydt )
    ax1.set_title( 'quiver' )
    ax2.streamplot( X , Y , dxdt , dydt )
    ax2.set_title( 'streamplot' )
    for ax in (ax1 , ax2):
        ax.set_xlim( 0 , 3 )
        ax.set_ylim( 0 , 10 )
        ax.set_xlabel( 'x (ADP)' )
        ax.set_ylabel( 'y (F6P)' )
    fig.tight_layout()
    fig.savefig( 'selkov_vector_field.png' , dpi=150 )


def plot_overlay():
    # 生の矢印は x, y が大きいところで長くなりすぎるので，向きだけを描く。
    # 縦横の目盛りが違うので，軸の長さで割ってから長さをそろえ，angles='xy' で軌跡の接線に合わせる
    N = 20
    x_max , y_max = 3 , 4
    [X, Y] = np.meshgrid( np.linspace( 0, x_max, N ) , np.linspace( 0, y_max, N ) )
    dxdt , dydt = selkov_rhs( X , Y )
    norm = np.hypot( dxdt / x_max , dydt / y_max ) / 0.04
    U , V = dxdt / norm , dydt / norm

    t_eval = np.linspace( 0 , 100 , 10000 )
    sol = solve_ivp( selkov_model , [0 , 100] , [1.0 , 1.0] , t_eval=t_eval , method='RK45' )

    x = np.linspace( 0 , 3 , 300 )
    plt.figure( figsize=(6, 4.8) )
    plt.quiver( X , Y , U , V , color='0.6' , angles='xy' , scale_units='xy' , scale=1 , width=0.003 )
    plt.plot( sol.y[0] , sol.y[1] , 'k-' , label='trajectory' )
    plt.plot( x , x / ( a + x**2 ) , 'r-' , label='dx/dt = 0' )
    plt.plot( x , b / ( a + x**2 ) , 'b-' , label='dy/dt = 0' )
    plt.plot( b , b / ( a + b**2 ) , 'ko' , fillstyle='none' , label='fixed point' )
    plt.xlim( 0 , x_max )
    plt.ylim( 0 , y_max )
    plt.xlabel( 'x (ADP)' )
    plt.ylabel( 'y (F6P)' )
    plt.legend( loc='upper right' )
    plt.tight_layout()
    plt.savefig( 'selkov_overlay.png' , dpi=150 )


plot_vector_fields()
plot_overlay()
plt.show()
