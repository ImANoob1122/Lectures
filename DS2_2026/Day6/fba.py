from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.linalg import null_space
from scipy.optimize import OptimizeResult, linprog

OUT_DIR: Final = Path(__file__).resolve().parent

# 演習1: 行 S1..S4，列 v1..v6
N_EX1: Final = sp.Matrix([
    [1, -1,  0,  0, -1,  0],
    [0,  1, -1,  0,  0,  0],
    [0,  0,  0,  0,  1, -1],
    [0,  0,  1, -1,  0,  1],
])
N_EX2: Final = sp.Matrix([[1, 1, 1]])

# 演習3: 変数は (v1, v2, b2)。b1 は右辺に置き，その双対値を shadow price として読む
B1: Final = 10.0
V_MAX: Final = (8.0, 6.0)
Z_COEF: Final = (40.0, 3.5)  # Z = 40 v_ATP + 3.5 v_NADH, v_ATP = v1, v_NADH = v2


def report_nullspace(name: str, n: sp.Matrix) -> None:
    kernel = n.nullspace()
    print(f'{name}: rank = {n.rank()}, 反応数 = {n.cols}, dim K = {len(kernel)}')
    for k in kernel:
        assert n * k == sp.zeros(n.rows, 1)
        print('  K の列:', list(k))


def check_integer_basis_ex2() -> None:
    k = sp.Matrix([[1, 1], [-1, 0], [0, -1]])
    assert N_EX2 * k == sp.zeros(1, 2) and k.rank() == 2
    print('演習2 の整数基底 K =', k.tolist(), '（N K = 0，rank 2）')
    print('  scipy の正規直交基底:\n', np.round(null_space(np.array(N_EX2, dtype=float)), 4))


def solve_fba(b1: float) -> OptimizeResult:
    c = [-Z_COEF[0], -Z_COEF[1], 0.0]  # linprog は最小化なので -Z
    a_eq = [[1.0, 1.0, 0.0],    # A: v1 + v2 = b1
            [1.0, 1.0, -1.0]]   # B: v1 + v2 - b2 = 0
    bounds = [(0.0, V_MAX[0]), (0.0, V_MAX[1]), (0.0, None)]
    return linprog(c, A_eq=a_eq, b_eq=[b1, 0.0], bounds=bounds, method='highs')


def report_fba() -> None:
    res = solve_fba(B1)
    if res.status != 0:
        raise RuntimeError(f'LP failed: {res.message}')
    v1, v2, b2 = res.x
    print(f'演習3: v1 = {v1:g}, v2 = {v2:g}, b2 = {b2:g}, Z = {-res.fun:g}')
    # eqlin.marginals[0] = d(-Z)/d b1 = -dZ/d b1 = 講義の pi
    print(f'演習4: pi(b1) = {res.eqlin.marginals[0]:g}（HiGHS の双対値）')
    lo, hi = solve_fba(B1 - 0.5), solve_fba(B1 + 0.5)
    print(f'  差分による検算 -dZ/db1 = {hi.fun - lo.fun:g}')
    print('  上限制約の双対値（v1, v2, b2）:', res.upper.marginals)


def plot_graphical() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.axvline(V_MAX[0], color='0.4', lw=1)
    ax.axhline(V_MAX[1], color='0.4', lw=1)
    v1 = np.linspace(0, 11, 2)
    ax.plot(v1, B1 - v1, color='0.6', lw=1)
    ax.plot([B1 - V_MAX[1], V_MAX[0]], [V_MAX[1], B1 - V_MAX[0]], 'k', lw=3, label='feasible set')
    for z in (250, 327, 400):
        ax.plot(v1, (z - Z_COEF[0] * v1) / Z_COEF[1], 'k:', lw=1)
        ax.text((z - Z_COEF[1] * 10.3) / Z_COEF[0] + 0.1, 10.3, f'$Z={z}$', fontsize=8,
                bbox=dict(fc='white', ec='none', pad=0.5))
    ax.plot(V_MAX[0], B1 - V_MAX[0], 'o', color='tab:orange', ms=9, zorder=5, label='optimum (8, 2)')
    ax.text(V_MAX[0] + 0.15, V_MAX[1] + 0.2, '$v_1 = 8$', fontsize=9)
    ax.text(0.2, V_MAX[1] + 0.2, '$v_2 = 6$', fontsize=9)
    ax.text(0.5, B1 - 0.3, '$v_1 + v_2 = 10$', fontsize=9)
    ax.set(xlim=(0, 11), ylim=(0, 11), xlabel='$v_1$ (mmol/gDW/h)', ylabel='$v_2$ (mmol/gDW/h)')
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fba_graphical.png', dpi=200)


def plot_robustness() -> None:
    b1_grid = np.linspace(0, sum(V_MAX), 141)
    z_opt = [-solve_fba(b).fun for b in b1_grid]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(b1_grid, z_opt, 'k', lw=2)
    ax.axvline(B1, color='tab:orange', ls='--', lw=1)
    ax.plot(B1, -solve_fba(B1).fun, 'o', color='tab:orange', ms=8)
    ax.text(4.8, 90, r'slope $= 40$ ($\pi = -40$)', fontsize=9)
    ax.text(10.2, 295, r'slope $= 3.5$' + '\n' + r'($\pi = -3.5$)', fontsize=9)
    ax.axvspan(sum(V_MAX), 15, color='0.9')
    ax.text(14.1, 150, 'infeasible', rotation=90, fontsize=8)
    ax.set(xlim=(0, 15), ylim=(0, 360), xlabel='$b_1$ (mmol/gDW/h)', ylabel='optimal $Z$')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fba_robustness.png', dpi=200)


def main() -> None:
    report_nullspace('演習1', N_EX1)
    report_nullspace('演習2', N_EX2)
    check_integer_basis_ex2()
    report_fba()
    plot_graphical()
    plot_robustness()


if __name__ == '__main__':
    main()
