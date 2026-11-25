import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

CONDITIONS = [ ( 'Condition 1' , 0.14 , 0.6 ) , ( 'Condition 2' , 0.06 , 0.6 ) ]
INIT = [ 1.0 , 1.0 ]
T_END = 100

x, y, a, b = sp.symbols("x y a b", real=True, positive=True)
f = -x + a * y + x**2 * y
g = b - a * y - x**2 * y
fix_x , fix_y = sp.solve( [ f , g ] , ( x , y ) )[0]
J_sym = sp.Matrix( [ [ f.diff( x ) , f.diff( y ) ] , [ g.diff( x ) , g.diff( y ) ] ] )
J_fix = sp.simplify( J_sym.subs( [ ( x , fix_x ) , ( y , fix_y ) ] ) )
J_func = sp.lambdify( ( a , b ) , J_fix , 'numpy' )


def eigenvalues( a_val , b_val ):
    return np.linalg.eigvals( np.array( J_func( a_val , b_val ) , dtype=float ) )


def hopf_point( b_val ):
    # trace = 0 は固有値の実部 = 0、つまり虚軸を横切る点
    roots = sp.solve( J_fix.trace().subs( b , b_val ) , a )
    return [ float( r ) for r in roots ]


def selkov_model( t , init , a_val , b_val ):
    x_val , y_val = init
    v2 = a_val * y_val + x_val**2 * y_val
    return [ v2 - x_val , b_val - v2 ]


def simulate( a_val , b_val , t_end ):
    t_eval = np.linspace( 0 , t_end , 100 * t_end + 1 )
    return solve_ivp( selkov_model , [ 0 , t_end ] , INIT , t_eval=t_eval , args=( a_val , b_val ) ,
                      method='RK45' , rtol=1e-9 , atol=1e-12 )


def peak_to_peak( sol , t_from , t_to ):
    mask = ( sol.t >= t_from ) & ( sol.t <= t_to )
    return np.ptp( sol.y[0][mask] ) , np.ptp( sol.y[1][mask] )


def plot_condition( ax_time , ax_phase , title , a_val , b_val ):
    sol = simulate( a_val , b_val , T_END )
    ax_time.plot( sol.t , sol.y[0] , label='ADP (x)' )
    ax_time.plot( sol.t , sol.y[1] , label='F6P (y)' )
    ax_time.set_title( f'{title}: a={a_val}, b={b_val}' )
    ax_time.set_xlabel( 't' )
    ax_time.set_ylabel( 'concentration' )
    ax_time.set_ylim( 0 , 1.3 * sol.y.max() )
    ax_time.legend( loc='upper right' , ncol=2 )

    xs = np.linspace( 0.05 , 2.5 , 300 )
    ax_phase.plot( sol.y[0] , sol.y[1] , 'k-' , lw=0.8 )
    ax_phase.plot( xs , xs / ( a_val + xs**2 ) , 'r-' , label='dx/dt = 0' )
    ax_phase.plot( xs , b_val / ( a_val + xs**2 ) , 'b-' , label='dy/dt = 0' )
    ax_phase.plot( b_val , b_val / ( a_val + b_val**2 ) , 'ko' , fillstyle='none' , label='fixed point' )
    ax_phase.set_xlim( 0 , 2.5 )
    ax_phase.set_ylim( 0 , 4 )
    ax_phase.set_xlabel( 'x (ADP)' )
    ax_phase.set_ylabel( 'y (F6P)' )
    ax_phase.legend( loc='upper right' )


def plot_eigen_path( ax , b_val ):
    # 条件1 (a=0.14) から条件2 (a=0.06) へ a を下げたときの固有値の動き
    a_path = np.linspace( 0.14 , 0.06 , 81 )
    eigs = np.array( [ eigenvalues( a_val , b_val ) for a_val in a_path ] )
    for k in range( 2 ):
        ax.plot( eigs[:, k].real , eigs[:, k].imag , 'k.' , ms=3 )
    for _, a_val , _ in CONDITIONS:
        ev = eigenvalues( a_val , b_val )
        ax.plot( ev.real , ev.imag , 'o' , ms=8 , fillstyle='none' , label=f'a={a_val}' )
    ax.axvline( 0 , color='gray' , lw=0.8 )
    ax.axhline( 0 , color='gray' , lw=0.8 )
    ax.set_xlabel( 'Re $\\lambda$' )
    ax.set_ylabel( 'Im $\\lambda$' )
    ax.set_title( f'eigenvalues, a: 0.14 $\\to$ 0.06 (b={b_val})' )
    ax.legend( loc='center right' )


def report():
    print( 'J at fixed point =' , J_fix )
    for title , a_val , b_val in CONDITIONS:
        ev = eigenvalues( a_val , b_val )
        J = np.array( J_func( a_val , b_val ) , dtype=float )
        print( f'{title} a={a_val} b={b_val}: trace={np.trace( J ):.6f} det={np.linalg.det( J ):.6f} eig={ev}' )
        sol = simulate( a_val , b_val , 300 )
        for t_from , t_to in [ ( 0 , 20 ) , ( 80 , 100 ) , ( 280 , 300 ) ]:
            px , py = peak_to_peak( sol , t_from , t_to )
            print( f'  ptp t={t_from}-{t_to}: x {px:.4f}, y {py:.4f}' )
    print( 'Hopf point a (b=0.6):' , hopf_point( 0.6 ) )


report()

fig , axes = plt.subplots( 2 , 2 , figsize=(10, 8) )
for col , ( title , a_val , b_val ) in enumerate( CONDITIONS ):
    plot_condition( axes[0, col] , axes[1, col] , title , a_val , b_val )
fig.tight_layout()
fig.savefig( 'selkov_hopf.png' , dpi=150 )

fig2 , ax2 = plt.subplots( figsize=(5, 4) )
plot_eigen_path( ax2 , 0.6 )
fig2.tight_layout()
fig2.savefig( 'selkov_hopf_eigen.png' , dpi=150 )
plt.show()
