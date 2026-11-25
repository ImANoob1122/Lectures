import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import re
from matplotlib.lines import Line2D

# 変数と定数
x, y, a, b = sp.symbols("x y a b", real=True, positive=True)

f = -x + a * y + x**2 * y
g =  b - a * y - x**2 * y

# ヌルクラインと固定点
solutions = sp.solve( [ -x + a * y + x**2 * y , b - a * y - x**2 * y ]  , ( x , y ) )
fix_x , fix_y = solutions[0]

# ヤコビ行列
J_sym = sp.Matrix( [ [ f.diff( x ) , f.diff( y ) ] , [ g.diff( x ) , g.diff( y ) ] ] )

# 相図を描く
a_grid = np.linspace( 0, 0.15, 15 )
b_grid = np.linspace( 0, 1.0, 10 )
[A, B] = np.meshgrid( a_grid , b_grid )
plt.figure( figsize=( 8 , 5 ) )

for a_val , b_val in zip( A.flatten() , B.flatten() ):
    J_num = J_sym.subs( [ ( x , fix_x ) , ( y , fix_y ) , ( a , a_val ), ( b , b_val ) ] )
    eig1, eig2 = J_num.eigenvals().keys()

    if eig1.is_real : # eig1 が実数ならば if 節に入る
        if eig1 < 0 and eig2 < 0 : # 実数・安定
            plt.plot( a_val , b_val , 'ko' ) 
        else : # 実数・不安定
            plt.plot( a_val , b_val , 'ko' , fillstyle='none') 
    else :
        if re( eig1 ) < 0 : # eig1 の実部が負であれば if 節に入る。すなわち複素数・安定の場合。
            plt.plot( a_val , b_val , 'ro' ) 
        else : # 複素数・不安定
            plt.plot( a_val , b_val , 'ro' , fillstyle='none') 

# 振動するパラメータと振動しないパラメータの位置関係を表示
oscillation_a = 0.06
oscillation_b = 0.6
stable_a = 0.14
plt.plot( oscillation_a , oscillation_b , 'b+', ms=20)
plt.plot( stable_a , oscillation_b , 'g+', ms=20)

# trace = 0（固有値の実部 = 0）を a について解くと Hopf 分岐の曲線になる
hopf_a = sp.solve( sp.simplify( J_sym.subs( [ ( x , fix_x ) , ( y , fix_y ) ] ).trace() ) , a )
hopf_func = sp.lambdify( b , hopf_a[0] , 'numpy' )
b_line = np.linspace( 0 , 1.0 , 200 )
plt.plot( hopf_func( b_line ) , b_line , 'b--' , lw=1 )

legend_items = [ Line2D( [] , [] , marker='o' , color=c , ls='none' , fillstyle=fs , label=label )
                 for c , fs , label in [ ( 'k' , 'full' , 'real, stable' ) , ( 'k' , 'none' , 'real, unstable' ) ,
                                         ( 'r' , 'full' , 'complex, stable' ) , ( 'r' , 'none' , 'complex, unstable' ) ] ]
legend_items.append( Line2D( [] , [] , color='b' , ls='--' , label='Hopf (trace = 0)' ) )
plt.legend( handles=legend_items , loc='center left' , bbox_to_anchor=( 1.02 , 0.5 ) )
plt.xlabel( 'a' )
plt.ylabel( 'b' )
plt.tight_layout()
plt.savefig( 'selkov_phase_diagram.png' , dpi=150 )
plt.show()

