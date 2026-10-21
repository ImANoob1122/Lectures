import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.1
BETA = 10
RHO = 25
X_OUTS = [ 0 , 5 , 10 ]

x = np.linspace( 0 , 10 , 500 )
fig , ( ax_fb , ax_hill ) = plt.subplots( 1 , 2 , figsize=( 11 , 4.2 ) )

# beta = 0: dx/dt = 0 は y によらず x = alpha * x_out の縦線になる
ax_fb.plot( x , x**2 / ( RHO + x**2 ) , 'b-' , label='dy/dt = 0' )
for x_out in X_OUTS:
    ax_fb.axvline( ALPHA * x_out , color='r' , linestyle='--' )
ax_fb.plot( [] , [] , 'r--' , label='dx/dt = 0 (X_out = 0, 5, 10)' )
ax_fb.set_xlim( 0 , 10 )
ax_fb.set_ylim( 0 , 1 )
ax_fb.set_title( 'no positive feedback (beta = 0)' )

# n = 1: dy/dt = 0 が上に凸になり、直線との交点は 1 つだけ
ax_hill.plot( x , x / ( RHO + x ) , 'b-' , label='dy/dt = 0' )
for x_out in X_OUTS:
    ax_hill.plot( x , ( x - ALPHA * x_out ) / BETA , 'r--' )
ax_hill.plot( [] , [] , 'r--' , label='dx/dt = 0 (X_out = 0, 5, 10)' )
ax_hill.set_xlim( 0 , 10 )
ax_hill.set_ylim( 0 , 0.4 )
ax_hill.set_title( 'no cooperativity (n = 1)' )

for ax in ( ax_fb , ax_hill ):
    ax.set_xlabel( 'x (TMG_in)' )
    ax.set_ylabel( 'y' )
    ax.legend( loc='lower right' )

fig.tight_layout()
fig.savefig( 'phaseplane_variants.png' , dpi=150 )
plt.show()
