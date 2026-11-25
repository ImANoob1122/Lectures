import sympy as sp

# 変数と定数
x = sp.Symbol('x')
y = sp.Symbol('y')

a = 0.06
b = 0.6

# 交点の座標を求める
solutions = sp.solve( [ x/(a+x**2) - y , b/(a+x**2) - y ] , ( x , y ) )

print(solutions)

