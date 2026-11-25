from fractions import Fraction
from math import log2

# 演習1の表: (ERK, CREB) の蛍光強度
CELLS = [ ( 0.3 , 5.8 ) , ( 1.8 , 17.9 ) , ( 1.8 , 9.3 ) , ( 0.8 , 11.2 ) ,
          ( 2.6 , 27.5 ) , ( 0.2 , 6.7 ) , ( 1.9 , 19.2 ) , ( 1.8 , 23.2 ) ]
ERK_BINS = [ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 3 ) ]
CREB_BINS = [ ( 0 , 10 ) , ( 10 , 20 ) , ( 20 , 30 ) ]
LOG2_3_APPROX = 1.6  # 演習5-1 の指示


def bin_index( value , bins ):
    for i , ( lo , hi ) in enumerate( bins ):
        if lo <= value < hi:
            return i
    raise ValueError( f'{value} is outside {bins}' )


def frequency_table( cells ):
    table = [ [ 0 ] * len( ERK_BINS ) for _ in CREB_BINS ]
    for erk , creb in cells:
        table[ bin_index( creb , CREB_BINS ) ][ bin_index( erk , ERK_BINS ) ] += 1
    return table


def approx_log2( p ):
    # p = 2^i 3^j の形のときだけ log2 3 = 1.6 で近似できる
    exponents = []
    for n in ( p.numerator , p.denominator ):
        i = j = 0
        while n % 2 == 0:
            n , i = n // 2 , i + 1
        while n % 3 == 0:
            n , j = n // 3 , j + 1
        if n != 1:
            raise ValueError( f'{p} is not of the form 2^i 3^j' )
        exponents.append( i + j * LOG2_3_APPROX )
    return exponents[0] - exponents[1]


def entropy( probs , log ):
    return sum( -p * log( p ) for p in probs if p > 0 )


def show( name , value_exact , value_approx ):
    print( f'{name:16s} exact {value_exact:.4f}   (log2 3 = 1.6) {value_approx:.4f}' )


counts = frequency_table( CELLS )
n = len( CELLS )
joint = [ [ Fraction( c , n ) for c in row ] for row in counts ]
p_erk = [ sum( joint[i][j] for i in range( len( CREB_BINS ) ) ) for j in range( len( ERK_BINS ) ) ]
p_creb = [ sum( row ) for row in joint ]

print( 'counts (rows: CREB 0-10, 10-20, 20-30 / cols: ERK 0-1, 1-2, 2-3)' )
for row in counts:
    print( '  ' , row )
print( 'joint' , [ [ str( p ) for p in row ] for row in joint ] )
print( 'P(ERK)' , [ str( p ) for p in p_erk ] , ' P(CREB)' , [ str( p ) for p in p_creb ] )
for name , probs in [ ( 'ERK' , p_erk ) , ( 'CREB' , p_creb ) ]:
    print( f'I({name})' , [ f'{-log2( p ):.4f} / {-approx_log2( p ):.2f}' for p in probs ] )

flat = [ p for row in joint for p in row ]
h_erk = ( entropy( p_erk , log2 ) , entropy( p_erk , approx_log2 ) )
h_creb = ( entropy( p_creb , log2 ) , entropy( p_creb , approx_log2 ) )
h_joint = ( entropy( flat , log2 ) , entropy( flat , approx_log2 ) )
show( 'H(ERK)' , *h_erk )
show( 'H(CREB)' , *h_creb )
show( 'H(ERK,CREB)' , *h_joint )
show( 'I(CREB;ERK)' , h_creb[0] + h_erk[0] - h_joint[0] , h_creb[1] + h_erk[1] - h_joint[1] )
